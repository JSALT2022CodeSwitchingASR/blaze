# SPDX-License-Identifier: MIT

module ARPALanguageModel

export parsefile

struct InvalidARPAFormatError
    lineno
    msg
end

Base.showerror(io::IO, e::InvalidARPAFormatError) =
    print(io, "line ", e.lineno, ": $(e.msg)")

@enum ParserState comment header body finished

"""
    parse(io | string)

Parse an ARPA-formatted `io` object/string and return a array of
dictionary for the ngrams.
"""
function Base.parse(str::Union{AbstractString, IO})
    state = comment

    ngram_counts = []
    ngrams = []

    lineno = 0
    for line in eachline(str)
        lineno += 1
        stripped_line = strip(line)

        # Ignore empty lines.
        if isempty(stripped_line)
            continue
        end


        if state == comment
            if startswith(stripped_line, "\\data\\")
                state = header
                continue
            end
        elseif state == header
            expected_order = length(ngram_counts) + 1
            if startswith(stripped_line, "\\$(length(ngrams) + 1)-grams:")
                state = body
                push!(ngrams, Dict())
                continue
            elseif startswith(stripped_line, "ngram $expected_order=")
                count = parse(Int, split(stripped_line, "=")[2])
                push!(ngram_counts, count)
            else
                throw(InvalidARPAFormatError(lineno, "unexpected ngram count: $stripped_line"))
            end
        elseif state == body
            if startswith(stripped_line, "\\end\\")
                state = finished
                break
            elseif startswith(stripped_line, "\\$(length(ngrams) + 1)-grams:")
                push!(ngrams, Dict())
                continue
            else
                tokens = split(stripped_line, "\t")
                if 2 <= length(tokens)  <= 3
                    ngram = tuple(split(tokens[2])...)
                    record = (
                        parse(Float32, tokens[1]),
                        length(tokens) == 3 ? parse(Float32, tokens[3]) : missing
                    )
                    ngrams[end][ngram] = record
                else
                    throw(InvalidARPAFormatError(lineno, "invalid ngram entry: $stripped_line"))
                end
            end
        end
    end

    if state != finished
        throw(InvalidARPAFormatError(lineno, "unexpected end of file"))
    end
    ngrams
end

function parsefile(path::AbstractString)
    open(path, "r") do f
        parse(f)
    end
end

end
