module MMIPreparation
using MarkovModels

using Semirings
using SparseArrays

function statemap(fsa, numpdf)
    sparse(
        1:nstates(fsa)+1,
        map(x -> x[end], [val.(fsa.λ)..., (numpdf+1,)]),
        one(eltype(fsa.α)),
        nstates(fsa) + 1,
        numpdf + 1
    )
end


function LinearFSA(K::Type{<:LogSemiring}, seq; silword,
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

end

