

function multinomial_powers(n_variables, max_order)
    # NOTE: n_variables should NOT include constant
    @show n_variables
    @show max_order
    Iterators.flatten(map(1:(max_order+1)) do order
        overflow_size = n_variables + order-1
        map(combinations(0:overflow_size-1, n_variables-1)) do indices
            starts = [0, (indices .+ 1)...]
            stops = [indices..., overflow_size]
            stops .- starts
        end
    end)
end
function multinomial(bases, powers)
    prod(bases .^ powers)
end


function multinomial_recombination(data::AbstractMatrix{T}, data_description, P::Int) where T
    """
    You might be tempted to apply a list of transforms arbitrarily, but 
    then you might end up with equivalent transforms, so that the columns
    are not independent.

    Build a matrix with columns representing multinomials up to degree P
    of all columns in data.

    input:
        data: column 1 is U; remaining columns are transforms of U including
              derivatives
        data_description: labels for each column
        P: maximum polynomial order of resulting columns
    output: 
        multinomials: multinomial combinations of columns in input data
        multinomials_description: descriptions of columns in multinomials
    """
    len_vars, n_vars = size(data)
    l_powers = multinomial_powers(n_vars, P) |> collect # don't power constant column
    n_powers = length(l_powers)
    multinomial_data = Matrix{T}(undef, len_vars, n_powers)
    multinomial_data_description = Vector{AbstractDescription}(undef, n_powers)
    for (i_power, powers) in enumerate(l_powers)
        for i_row in 1:len_vars
            multinomial_data[i_row,i_power] = multinomial(data[i_row,:], powers')
        end
        multinomial_data_description[i_power] = multinomial(data_description, powers)
    end

    return multinomial_data, multinomial_data_description
end


