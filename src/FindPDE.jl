module FindPDE

include("differencing.jl")
include("multinomials.jl")

function build_linear_system(arr_frames::U, dt, dx, derivative_order, polynomial_order) where
    {T, SpaceArr<:AbstractVector{T}, U<:AbstractVector{SpaceArr}}
    # TODO: multidimensional space
    n_t = length(arr_frames)
    n_x = length(arr_frames[1])
    
    matrix_u::Matrix{T} = hcat(arr_frames...) # FIXME: if N_SPACE > 1, need to unroll before splatting 
    SPACE_DIM = 1
    TIME_DIM = 2

    # Matrix where each column is a flattened derivative matrix
    # Col 1: u
    # Cols 2-(derivative_order+1): u_(x...)
    n_space_derivatives = derivative_order + 1
    space_derivatives = Matrix{T}(undef, prod(size(matrix_u)), n_space_derivatives) 

    all_derivatives[:,1] = matrix_u[:]

    time_derivative_order = 1
    time_derivative_approximation_order = 2
    u_t = finite_diff(matrix_u, dt, TIME_DIM, time_derivative_order, 
                      time_derivative_approximation_order)[:]

    for order in 1:derivative_order
        all_derivatives[:,order+2] = finite_diff(matrix_u, dx, SPACE_DIM, order)
    end
   
    Φ, desc =  multinomial_recombination(all_derivatives, nothing, polynomial_order) 
end


end # module
