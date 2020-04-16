
function finite_diff(u::AbstractArray{T}, dx, dim, derivative_order, approximation_order=2) where T
    # TODO: Test against FiniteDiff in PDE-FIND
    derivative_operator = CenteredDifference{dim}(derivative_order, approximation_order, dx, length(u))
    boundary_operator = Dirichlet0BC(T) #FIXME: sends boundary to zero
    return derivative_operator * boundary_operator * u
end
