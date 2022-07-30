# SPDX-License-Identifier: MIT

module LanguageModels

using MarkovModels

export LanguageModelFSA

const _default_startsym = "<s>"
const _default_stopsym = "</s>"


function _extract_initweights!(K, ngrams; startsym = _default_startsym)

    _, global_initweight = pop!(ngrams, (startsym,))

    # Remove ngrams starting by `startsym` and longer than 2.
    filter!(x -> x.first[1] != startsym || length(x.first) <= 2, ngrams)

    initweights = Dict()
    for ngram in filter(x -> x[1] == startsym, keys(ngrams))
		weight, _ = pop!(ngrams, ngram)
        initweights[ngram[2:end]] = K(weight)
	end

    K(global_initweight), initweights
end


function _extract_finalweights!(K, ngrams; stopsym = _default_stopsym)

    global_finalweight, _ = pop!(ngrams, (stopsym,))

    finalweights = Dict()
    for ngram in filter(x -> x[end] == stopsym, keys(ngrams))
		weight, _ = pop!(ngrams, ngram)
        finalweights[ngram[1:end-1]] = K(weight)
	end

    K(global_finalweight), finalweights
end


struct EpsilonState{T}
    context::T
end
EpsilonState(s::EpsilonState) = EpsilonState(s.context)
Base.:(==)(s1::EpsilonState, s2::EpsilonState) = s1.context == s2.context
Base.hash(x::EpsilonState, h::UInt) = hash(x.context, h)


function _get_src_dest(order, state)
	src = state[1:end-1]
    src = length(src) == 0 ? EpsilonState(src) : src
	dest = state[max(1, length(state) - (order - 1) + 1):end]
	src, dest
end


function _get_arcs(K, order, ngrams)
	arcs = Dict()
	eps = Set()
	for (ngram, (weight, backoff)) in ngrams
		src, dest = _get_src_dest(order, ngram)
		if ! ismissing(backoff)
            src = EpsilonState(src)
		    arcs[(src, dest)] = weight
		    push!(eps, src)

            src = length(dest) == order - 1 ? dest : EpsilonState(dest)
            dest = EpsilonState(dest[2:end])

			arcs[(src, dest)] = backoff
			push!(eps, dest)
        else
            arcs[(src, dest)] = weight
            #src, dest = dest, EpsilonState(dest[2:end])
		end
	end

    # Connect each emitting state to its "epsilon" counter part.
	for e in eps
        if length(e.context) > 0
            arcs[e.context, e] = one(K)
        end
	end

	arcs
end


function _get_states(initweights, arcs, finalweights)
	istates = Set(keys(initweights))
	fstates = Set(keys(finalweights))
	srcstates = Set([x[1] for x in keys(arcs)])
	deststates = [x[2] for x in keys(arcs)]

	allstates = istates ∪ fstates ∪ srcstates ∪ deststates
    emitting_states = filter(x -> ! (x isa EpsilonState), allstates)
    eps_states = filter(x -> x isa EpsilonState, allstates)

    emitting_states, eps_states
end


function _get_state_ids(states, eps_states)
    state2id = Dict()
	id2state = Dict()
	neps = 0
	nstates = 0
	for s in (states ∪ eps_states)
		if s isa LanguageModels.EpsilonState
			if length(s.context) == 0
				state2id[s] = 0
				id2state[0] = s
			else
				neps += 1
				state2id[s] = -neps
				id2state[-neps] = s
			end
		else
			nstates += 1
			state2id[s] = nstates
			id2state[nstates] = s
		end
	end
	state2id, id2state
end


function LanguageModelFSA(K, raw_ngrams)
    # Convert the raw n-grams to the semiring `K`.
    ngrams = Dict()
    for (ngram, (w, b)) in raw_ngrams
        ngrams[ngram] = (K(w), ismissing(b) ? missing : K(b))
    end

    # Check the first element to get the semiring.
    K = typeof(first(ngrams).second[1])

    # Get the order of the langauge model by finding the longest n-gram.
    order = maximum(x -> length(x.first), ngrams)

    # Extract first the initial/final weights.
	global_initweight, initweights = LanguageModels._extract_initweights!(K, ngrams)
	global_finalweight, finalweights = LanguageModels._extract_finalweights!(K, ngrams)

    # Extract the arcs.
	arcs = LanguageModels._get_arcs(K, order, ngrams)

    # Get the set of states from the arcs.
	states, eps_states = LanguageModels._get_states(initweights, arcs,
											    	finalweights)

    # Add the global initial weight to all 1-gram states.
    for s in filter(x -> length(x) == 1, states)
        initweights[s] = get(initweights, s, zero(K)) + global_initweight
    end

    # Add the global initial weight to all the states.
    for s in states
        finalweights[s] = get(finalweights, s, zero(K)) + global_finalweight
    end

    # Construct the mapping state <-> id.
	state2id, id2state = LanguageModels._get_state_ids(states, eps_states)

    # Construct the symbol table.
    symtable = [Label(join(id2state[i], "_")) for i in 1:length(states)]

    # Make the FSA.
    FSA(
        [state2id[s] => w for (s, w) in initweights],
        [(state2id[src], state2id[dest]) => w for ((src, dest), w) in arcs],
        [state2id[s] => w for (s, w) in finalweights],
        symtable
    )
end

end

