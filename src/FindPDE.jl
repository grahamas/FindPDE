module FindPDE

using DiffEqOperators
using DataFrames
using MLJLinearModels, MLJ
using LinearAlgebra
using Combinatorics
using LaTeXStrings
using PrettyPrinting

export build_linear_system

include("differencing.jl")
include("multinomials.jl")
include("descriptions.jl")
include("regression.jl")

function build_linear_system(matrix_u::Matrix{T}, dt, dx, derivative_order, polynomial_order) where
    {T}
    # TODO: multidimensional space
    n_x, n_t = size(matrix_u)
    
    SPACE_DIM = 1
    TIME_DIM = 2

    # Matrix where each column is a flattened derivative matrix
    # Col 1: u
    # Cols 2-(derivative_order+1): u_(x...)
    n_space_derivatives = derivative_order + 1
    space_derivatives = Matrix{T}(undef, prod(size(matrix_u)), n_space_derivatives) 
    space_derivatives_desc = Vector{AbstractDescription}(undef, n_space_derivatives)

    space_derivatives[:,1] = matrix_u[:]
    base_desc = AtomicDescription("u")
    space_derivatives_desc[1] = base_desc

    time_derivative_order = 1
    time_derivative_approximation_order = 2
    u_t = finite_diff(matrix_u', dt, time_derivative_order, 
                      time_derivative_approximation_order)[:]
    u_t_desc = differentiate(base_desc, :t)

    for order in 1:derivative_order
        space_derivatives[:,order+1] = finite_diff(matrix_u, dx, order)[:]
        space_derivatives_desc[order+1] = differentiate(base_desc, :x, order)
    end
   
    space_terms, space_terms_desc =  multinomial_recombination(space_derivatives, space_derivatives_desc, polynomial_order) 
    all_terms = hcat(u_t, space_terms)
    all_terms_desc = [u_t_desc, space_terms_desc...]
    all_terms_df = convert(DataFrame, all_terms)
    names!(all_terms_df, Symbol.(all_terms_desc))
    return all_terms_df
end


end # module
