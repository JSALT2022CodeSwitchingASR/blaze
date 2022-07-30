### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 91fd0a6e-0ced-11ed-3582-4d9adab28b6e
begin
	using Pkg
	Pkg.activate("../")

	using Semirings
	using MarkovModels
	using SparseArrays
	using Serialization

	using Revise
end

# ╔═╡ 269a0b84-7218-43db-8061-6e4e57510e6a
include("utils/arpa.jl")

# ╔═╡ 02c3d3f3-b19e-408e-b6e0-8ab10b6c908b
include("utils/lm.jl")

# ╔═╡ b61cc15a-ae7c-4269-a89b-9eda76ddae43
K = LogSemiring{Float32}

# ╔═╡ 09e2726f-1679-4f09-bc4e-d2ea7f1f0580
md"""
## Toy Example
"""

# ╔═╡ d8b3a5d0-b69f-4f2c-bf6d-0e69188845d1
md"""
``
y = \mathbf{a}x + b
``
"""

# ╔═╡ 698ee984-89c0-410b-983c-9839116e0d6d
order_toy, ngrams_toy = ARPA.parsefile("../test/lm.arpa")

# ╔═╡ 077e9ee8-70f8-42fe-ae48-9c07f5450b67
L_toy = LanguageModels.LanguageModelFSA(K, ngrams_toy)

# ╔═╡ 4bef34ea-60c4-4774-b6d4-9fb09b632c11
rmepsilon(L_toy)

# ╔═╡ 9daaf2de-2661-4f44-a27f-8f21c5eac03f
order, ngrams = ARPA.parsefile("../data/lm/xhosa/lm.3_gram.arpa")

# ╔═╡ 4670505f-e16f-497d-aa4f-aaf42dab217d
G = LanguageModels.LanguageModelFSA(K, ngrams);

# ╔═╡ 673ddb66-6b6d-45c1-a94b-c690c5edbef7
serialize(
	"../data/lm/xhosa/lm.3_gram.fsa",
	FSA(G.α, G.T, G.ω)
)

# ╔═╡ 4f823086-4784-4a0f-af10-471e2777ef51
hmms = deserialize("../data/graphs/xhosa/hmms.dictfsa")

# ╔═╡ a6bf9faa-9bab-4df5-b8aa-7859e96f88e1
lexicon = deserialize("../data/graphs/xhosa/lexicon.dictfsa");

# ╔═╡ 671ad7c7-c038-420f-b08c-e996672621bd
lexicon #["IYELENQE"]

# ╔═╡ 8bdc1c5e-c714-43a2-8906-f9f94fb93aed
GL = replace(G) do i
	context_word = val(G.λ[i])[1]
	word = split(context_word, "_")[end]
	lexicon[word]
end;

# ╔═╡ 1ccb3089-3505-4348-a002-c0af3a5031b6
GLH = replace(GL) do i
	hmms["t_h"]
end;

# ╔═╡ f8713989-ee7d-4f43-a3e4-cac531c2cebc
serialize(
	"../data/lm/xhosa/GLH.fsa",
	FSA(GLH.α, GLH.T, GLH.ω)
)

# ╔═╡ 28ded053-1d64-4d71-b6bf-25c934735282
serialize(
	"../data/lm/xhosa/GL.fsa",
	FSA(GL.α, GL.T, GL.ω)
)

# ╔═╡ 4bc94934-9064-4053-a661-774d687e14f2
size(GLH.T)

# ╔═╡ 1907f0d1-caac-4832-89b9-2ffea2cf9ee3
narcs(fsa::FSA{<:Semiring, <:AbstractSparseMatrix}) = nnz(fsa.T)

# ╔═╡ b12773a2-f5a5-48e5-97ac-0dcb816ba081
narcs(fsa::FSA{<:Semiring, <:MarkovModels.SparseLowRankMatrix}) = 
	nnz(fsa.T.S) + nnz(fsa.T.D) + nnz(fsa.T.U) + nnz(fsa.T.V)

# ╔═╡ 3bef8c64-d87b-4757-a79f-debe5a360884
narcs(GL)

# ╔═╡ 732852dd-0ca1-4610-b062-76c979c78fc1
narcs(G)

# ╔═╡ e6fe2811-3250-4d2c-8b63-4df5257fd7fd
typeof(GLH)

# ╔═╡ 21ea486c-4962-46b6-be01-54314f2b1b1c
typeof(GL.α)

# ╔═╡ Cell order:
# ╠═91fd0a6e-0ced-11ed-3582-4d9adab28b6e
# ╠═269a0b84-7218-43db-8061-6e4e57510e6a
# ╠═02c3d3f3-b19e-408e-b6e0-8ab10b6c908b
# ╠═b61cc15a-ae7c-4269-a89b-9eda76ddae43
# ╠═09e2726f-1679-4f09-bc4e-d2ea7f1f0580
# ╠═d8b3a5d0-b69f-4f2c-bf6d-0e69188845d1
# ╠═698ee984-89c0-410b-983c-9839116e0d6d
# ╠═077e9ee8-70f8-42fe-ae48-9c07f5450b67
# ╠═4bef34ea-60c4-4774-b6d4-9fb09b632c11
# ╠═9daaf2de-2661-4f44-a27f-8f21c5eac03f
# ╠═4670505f-e16f-497d-aa4f-aaf42dab217d
# ╠═673ddb66-6b6d-45c1-a94b-c690c5edbef7
# ╠═4f823086-4784-4a0f-af10-471e2777ef51
# ╠═a6bf9faa-9bab-4df5-b8aa-7859e96f88e1
# ╠═671ad7c7-c038-420f-b08c-e996672621bd
# ╠═8bdc1c5e-c714-43a2-8906-f9f94fb93aed
# ╠═1ccb3089-3505-4348-a002-c0af3a5031b6
# ╠═f8713989-ee7d-4f43-a3e4-cac531c2cebc
# ╠═28ded053-1d64-4d71-b6bf-25c934735282
# ╠═3bef8c64-d87b-4757-a79f-debe5a360884
# ╠═4bc94934-9064-4053-a661-774d687e14f2
# ╠═1907f0d1-caac-4832-89b9-2ffea2cf9ee3
# ╠═b12773a2-f5a5-48e5-97ac-0dcb816ba081
# ╠═732852dd-0ca1-4610-b062-76c979c78fc1
# ╠═e6fe2811-3250-4d2c-8b63-4df5257fd7fd
# ╠═21ea486c-4962-46b6-be01-54314f2b1b1c
