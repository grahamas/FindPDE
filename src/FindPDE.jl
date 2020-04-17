module FindPDE

using DiffEqOperators
using MLDataUtils
using LinearAlgebra
using Combinatorics
using LaTeXStrings

export build_linear_system

include("differencing.jl")
include("multinomials.jl")
include("descriptions.jl")

function build_linear_system(arr_frames::U, dt, dx, derivative_order, polynomial_order) where
    {T, SpaceArr<:AbstractVector{T}, U<:AbstractVector{SpaceArr}}
    # TODO: multidimensional space
    n_t = length(arr_frames)
    n_x = length(arr_frames[1])
    
    matrix_u::Matrix{T} = hcat(arr_frames...) # FIXME: if N_SPACE > 1, need to unroll before splatting 
    SPACE_DIM = 1
    TIME_DIM = 2

    # TODO: use e.g. AxisArrays for more efficiency
    # TODO: use a better name than Φ
    base_desc = AtomicDescription("u") 
    time_derivative_desc = differentiate(base_desc, :t)
    space_derivative_descs = [differentiate(base_desc, :x, order) for order in 1:derivative_order]
    term_descs = [base_desc, time_derivative_desc, space_derivative_descs...]
    Φ = DataFrame([T for _ term_descs], Symbol.(term_descs), n_t * n_x)

    Φ[!, Symbol(base_desc)] = matrix_u[:]

    time_derivative_order = 1
    time_derivative_approximation_order = 2
    Φ[!, Symbol(time_derivative_desc)] = finite_diff(matrix_u', dt, 
                                                     time_derivative_order,
                                                     time_derivative_approximation_order)[:]

    for order in 1:derivative_order
        local this_desc = differentiate(base_desc, :x, order)
        Φ[!, Symbol(this_desc)] = finite_diff(matrix_u, dx, order)[:]
    end
   
    Φ, Φ_desc =  multinomial_recombination(Φ, [base_desc, space_derivative_descs...], polynomial_order) 
end


end # module
