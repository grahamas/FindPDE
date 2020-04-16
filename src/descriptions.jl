
abstract type AbstractDescription end

struct AtomicDescription <: AbstractDescription
    name::String
end
Base.show(io::IO, desc::AtomicDescription) = print(io, desc.name)
### Powers
struct ExponentiatedDescription{D<:AbstractDescription} <: AbstractDescription
    base::D
    exponent::Int
end
function Base.:(^)(left::AbstractDescription, right::Int)
    ExponentiatedDescription(left, right)
end
Base.show(io::IO, desc::ExponentiatedDescription) = print(io, latexstring("\\left($(desc.base)\\right)^{$(desc.exponent)}"))

### Products
struct DescriptionProduct <: AbstractDescription
    multiplicands::Vector{<:AbstractDescription}
end
function Base.:(*)(left::DescriptionProduct, right::DescriptionProduct)
    DescriptionProduct([left.multiplicands..., right.multiplicands...])
end
function Base.:(*)(left::DescriptionProduct, right::AbstractDescription)
    DescriptionProduct([left.multiplicands..., right])
end
function Base.:(*)(left::AbstractDescription, right::DescriptionProduct)
    DescriptionProduct([left, right.multiplicands...])
end
Base.:(*)(left::AbstractDescription, right::AbstractDescription) = DescriptionProduct([left, right])
Base.show(io::IO, desc::DescriptionProduct) = print(io, latexstring("$(desc.multiplicands...)"))

### Derivatives
struct DerivativeDescription{D} <: AbstractDescription
    base::D
    derivatives::Dict{Symbol,Int}
end
function differentiate(desc::DerivativeDescription, sym::Symbol, order=1)
    derivatives = Dict(desc.derivatives)
    if haskey(derivatives, sym)
        derivatives[sym] += order
    else
        derivatives[sym] = order
    end
    return DerivativeDescription(desc.base, derivatives)
end
differentiate(desc::AbstractDescription, sym::Symbol, order=1) = DerivativeDescription(desc, Dict(sym => order))
function Base.show(io::IO, desc::DerivativeDescription)
    derivative_strs = ["D_{$(key)}^{$(val)}" for (key, val) in pairs(desc.derivatives)]
    print(io, latexstring("$(derivative_strs...)$(desc.base)"))
end
