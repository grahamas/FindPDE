
function finite_diff(u::AbstractArray{T}, dx, derivative_order, approximation_order=2) where T
    dim = 1
    # TODO: Test against FiniteDiff in PDE-FIND
    derivative_operator = CenteredDifference{dim}(derivative_order, approximation_order, dx, size(u, dim))
    boundary_operator = Dirichlet0BC(T) #FIXME: sends boundary to zero
    op = derivative_operator * boundary_operator
    return op * u
end
