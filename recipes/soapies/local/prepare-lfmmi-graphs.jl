
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("./")

using GZip
@everywhere using JSON
@everywhere using MarkovModels
using ProgressMeter
@everywhere using Semirings
@everywhere using Serialization
@everywhere using SparseArrays
using TOML


@everywhere function statemap(fsa, numpdf)
    sparse(
        1:nstates(fsa)+1,
        map(x -> x[end], [val.(fsa.λ)..., (numpdf+1,)]),
        one(eltype(fsa.α)),
        nstates(fsa) + 1,
        numpdf + 1
    )
end

@everywhere function LinearFSA(K::Type{<:LogSemiring}, seq; silword,
                              init_silprob = 0, silprob = 0,
				              final_silprob = 0)
	arcs = []

	if init_silprob > 0
		init = [1 => K(log(init_silprob)), 2 => K(log(1 - init_silprob))]
		push!(arcs, (1, 2) => one(K))
		labels = [Label(silword), Label(seq[1])]
		scount = 2
	else
		init = [1 => one(K)]
		labels = [Label(seq[1])]
		scount = 1
	end

	for (i, s) in enumerate(seq[2:end])
		if silprob > 0
			push!(arcs, (scount, scount + 1) => K(log(silprob)))
			push!(arcs, (scount, scount + 2) => K(log(1 - silprob)))
			push!(arcs, (scount + 1, scount + 2) => one(K))
			push!(labels, Label(silword))
			push!(labels, Label(s))
			scount += 2
		else
			push!(arcs, (scount, scount + 1) => one(K))
			push!(labels, Label(s))
			scount += 1
		end
	end

	if final_silprob > 0
		final = [scount => K(log(1 - final_silprob)),
				 scount + 1 => one(K)]
		push!(arcs, (scount, scount + 1) => K(log(final_silprob)))
		push!(labels, Label(silword))
	else
		final = [scount => one(K)]
	end

	FSA(init, arcs, final, labels)
end

function make_hmms(units, topojson)
	numpdf = 0
	jsondata = JSON.parsefile(topojson)
	nstates = length(jsondata["labels"])
	unitdict = Dict()
	open(units, "r") do f
        @showprogress for line in readlines(f)
			jsondata["labels"] = collect(numpdf+1:numpdf+nstates)
            token = split(line)[1]
			unitdict[String(token)] = FSA(json(jsondata))
			numpdf += nstates
		end
	end
	unitdict, numpdf
end

function make_lexicon(K, lexicon; silword)
	lfsa = Dict()

	open(lexicon, "r") do f
        @showprogress for line in readlines(f)
			tokens = String.(split(line))
			word, pronun = tokens[1], tokens[2:end]

			fsa = LinearFSA(K, pronun; silword)
			if word in keys(lfsa)
				lfsa[word] = union(lfsa[word], fsa) |> minimize |> renorm
			else
				lfsa[word] = fsa
			end
		end
	end
	lfsa
end

function make_numerator_graphs(K, folder, manifest, lexicon, hmms, numpdf;
                               silword, unkword, init_silprob, silprob,
                               final_silprob, ngram_order)

    @everywhere workers() mkpath(joinpath($folder, "$(myid() - 1)"))
    @everywhere workers() rm(joinpath($folder, "$(myid() - 1)", "fsa.scp"), force=true)
    @everywhere workers() rm(joinpath($folder, "$(myid() - 1)", "smap.scp"), force=true)

    fh = GZip.open(manifest)
    lines = readlines(fh)
    close(fh)

    ngrams = @showprogress @distributed (a, b) -> mergewith((x, y) -> x .+ y, a, b) for line in lines
        utterance = JSON.parse(line)
		uttid = utterance["id"]
        seq = split(utterance["text"])

		if isempty(seq)
		    Dict()
	    else
            seq = [s in keys(lexicon) ? s : unkword for s in seq]

            G = LinearFSA(K, seq; silword, init_silprob, silprob, final_silprob)
            GL = replace(i -> lexicon[ val(G.λ[i])[end] ], G)
            GLH = replace(i -> hmms[ val(GL.λ[i])[end] ], GL)
            fsa_path = joinpath(folder, "$(myid() - 1)", uttid * ".fsa")
            serialize(fsa_path, compile(GLH))
            smap_path = joinpath(folder, "$(myid() - 1)", uttid * ".smap")
            serialize(smap_path, statemap(GLH, numpdf))

            open(joinpath(folder, "$(myid() - 1)", "fsa.scp"), "a") do f
                println(f, uttid, " ", fsa_path)
            end
            open(joinpath(folder, "$(myid() - 1)", "smap.scp"), "a") do f
                println(f, uttid, " ", smap_path)
            end

            totalngramsum(GL, order = ngram_order)
        end
	end

	ngrams
end

# Look up the config file in the CONFIG environment variable.
config = TOML.parsefile(ENV["CONFIG"])

mkpath(config["supervision"]["outdir"])

@info "Make the HMMs..."
hmms, numpdf = make_hmms(config["data"]["units"], config["supervision"]["topo"])

open(joinpath(config["supervision"]["outdir"], "numpdf"), "w") do f
    println(f, "$numpdf")
end

# Extract the semiring type from the HMMs
K = eltype(collect(values(hmms))[1].α)

@info "Build the lexicon..."
lexicon = make_lexicon(K, config["data"]["lexicon"];
                       silword = config["supervision"]["silword"])


#@info "Build the numerator graphs (train) ($(Threads.nthreads()) threads)..."
@info "Build the numerator graphs (train) ($(nprocs() - 1) workers)..."
outfolder = joinpath(config["supervision"]["outdir"], "numfsas", "train")
mkpath(outfolder)
ngrams = make_numerator_graphs(
    K,
    outfolder,
    config["data"]["train_manifest"],
    lexicon,
    hmms,
    numpdf;
    silword = config["supervision"]["silword"],
    unkword = config["supervision"]["unkword"],
    init_silprob = config["supervision"]["initial_silprob"],
    silprob = config["supervision"]["silprob"],
    final_silprob = config["supervision"]["final_silprob"],
    ngram_order = config["supervision"]["ngram_order"]
)

open(joinpath(outfolder, "fsa.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "fsa.scp")))
    end
end

open(joinpath(outfolder, "smap.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "smap.scp")))
    end
end


@info "Build the numerator graphs (dev) ($(nprocs() - 1) workers)..."
outfolder = joinpath(config["supervision"]["outdir"], "numfsas", "dev")
mkpath(outfolder)
ngrams = make_numerator_graphs(
    K,
    outfolder,
    config["data"]["dev_manifest"],
    lexicon,
    hmms,
    numpdf;
    silword = config["supervision"]["silword"],
    unkword = config["supervision"]["unkword"],
    init_silprob = config["supervision"]["initial_silprob"],
    silprob = config["supervision"]["silprob"],
    final_silprob = config["supervision"]["final_silprob"],
    ngram_order = config["supervision"]["ngram_order"]
)

open(joinpath(outfolder, "fsa.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "fsa.scp")))
    end
end

open(joinpath(outfolder, "smap.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "smap.scp")))
    end
end

@info "Build the denominator graph..."
L = LanguageModelFSA(ngrams)
lmfsa = replace(i -> hmms[ val(L.λ[i])[end] ], L)
serialize(joinpath(config["supervision"]["outdir"], "denominator") * ".fsa",
          compile(lmfsa))
serialize(joinpath(config["supervision"]["outdir"], "denominator") * ".smap",
          statemap(lmfsa, numpdf))

