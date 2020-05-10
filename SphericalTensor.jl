# Dependencies

# using Pkg
# Pkg.add("WignerSymbols")
# Pkg.add("LinearAlgebra")
# Pkg.add("JuliennedArrays")
# Pkg.add("Statistics")
# Pkg.add("GeometryTypes")
# Pkg.add("Makie")
# Pkg.add("Colors")


using WignerSymbols
using LinearAlgebra
using JuliennedArrays
using Statistics
using GeometryTypes, Colors, Makie



# Our first step is to make spherical tensors semi-native. Note this only works with integer valued spherical harmonics
# This takes two steps.  First we define and indexer for the spherical harmonics (the MRange type)
# This is a special unit range type which defines the size of the tensor
# It signals to the compile that broadcasting must be initiated with another spherical tensor.
# Plus the methods for getting Base to understand how it works
# Then we define the spherical tensor type and methods for accessing it's index in the natural way



struct MRange <: AbstractUnitRange{Int}
    start::Int
    stop::Int
end

function MRange(L)
    MRange(-L,L)
end

# definition
struct Spherical{L} <: AbstractArray{Complex{Float64},1}
    x::Array{Complex{Float64},1}
end

# standard constructor
function Spherical(x)
    Spherical{Int((length(x)-1)/2)}(x)
end

# empty initialization
function Spherical{L}() where L
    Spherical(zeros(2*L+1))
end

# interface with Base
Base.axes(y::Spherical{L}) where L = (MRange(L),)
Base.size(y::Spherical{L}) where L = (2*L+1,)
function Base.getindex(y::Spherical{L}, m::Int) where L
    y.x[L+m+1]
end
function Base.setindex!(y::Spherical{L}, v ,m::Int) where L
    y.x[L+m+1] = v
end
function Base.similar(A::AbstractArray, T::Type, shape::Tuple{MRange,Vararg{MRange}})
    Spherical(A)
end
function Base.similar(f::Any, shape::Tuple{MRange,Vararg{MRange}}) 
    #something didn't work here from the recipe book, might be a bug
    Spherical(first(shape))
end

## Basic Operations on Spherical Tensors

# direct addition

import Base: +
function +(x::Spherical{L}, y::Spherical{L}) where L
    Spherical{L}(x.x .+ y.x)
end


# clebsch product into L3 irrep 
function cleb(y1::Spherical{L1},y2::Spherical{L2},L3::Int) where {L1, L2}
    if abs(L1 - L2) <= L3 <= L1 + L2
        return Spherical{L3}(
            [ sum( clebschgordan(L1, m1, L2, m3 - m1, L3, m3) * y1[m1] * y2[m3 - m1] 
                for m1 in -L1:L1 if abs(m3 - m1) <= L2)
            for m3 in -L3:L3 ] # comprehension and index work makes this easy!
         )
    else
        nothing
    end
end

using SparseArrays

# Generator matrices and rotations
# lowerJ |l,m> = sqrt(l(l+1) - m(m-1)) |l,m-1>
# raiseJ |l,m> = sqrt(l(l+1) - m(m+1)) |l,m+1>
function raiseJ(L)
    rows = L .+ 1 .- ((- L):(L-1))
    cols = L .+ 1 .- ((1-L):L)
    vals = sqrt.( L*(L+1) .- (-L:(L-1)) .* ((-L+1):L))
    sparse(rows,cols, vals, 2L+1, 2L+1)
end 

function lowerJ(L)
    rows = L .+ 1 .- ((1-L):L)
    cols = L .+ 1 .- ((- L):(L-1))
    vals = sqrt.( L*(L+1) .- (-L:(L-1)) .* ((-L+1):L))
    sparse(rows,cols, vals, 2L+1, 2L+1)
end 

function Jz(L)
    rows = L .+ 1 .+ (-L:L)
    cols = L .+ 1 .+ (-L:L)
    vals = (-L:L)
    sparse(rows,cols, vals, 2L+1, 2L+1)
end 
 
# the representation of jx and jy in irrep L
Jx(L) = (raiseJ(L) + lowerJ(L))/2
Jy(L) = (raiseJ(L) - lowerJ(L))/(2*im)

function rotate!(y::Spherical{L}, nvec::Array) where L 
    # rotates the spherical tensor around the axis defined by nvec
    # angle is given by the magnitude of nvec.
    mat = exp( Matrix(im .* (
        Jx(L) .* nvec[1] .+  
        Jy(L) .* nvec[2] .+ 
        Jz(L) .* nvec[3]) ))
    y = Spherical{L}(mat * y.x); #must be made dense to get this method to work
end

function rotate(y::Spherical{L}, nvec::Array) where L 
    # rotates the spherical tensor around the axis defined by nvec
    # angle is given by the magnitude of nvec.
    mat = exp( Matrix(im .* (
        Jx(L) .* nvec[1] .+  
        Jy(L) .* nvec[2] .+ 
        Jz(L) .* nvec[3]) ))
    return Spherical{L}(mat * y.x); 
    # must be made dense to use Base.exp method
    # a sparse matrix exponentiator from a package could speed this up,
    # not sure how much we will need this function (it's already fast)
end

function zAxisTo(new::Array)
    # return the smalles nvec that takes the z axis to point new
    b = normalize(new)
    if abs(b[3]) < 1 # have to deal with the degenerate case
        return normalize([-b[2], b[1], 0])*acos(b[3])
    else
        return [1, 0, 0]*acos(b[3])
    end
end

function harmonic(y::Spherical{L}, point::Array) where L
    # calculates the value of the spherical harmonic at point projected onto the unit sphere
    rotate(y::Spherical{L}, zAxisTo(point))[0]*sqrt((2L + 1)/(4π))
    # The m=0 component of the spherical tensor with point defining the z-axis
end

## Lets check normalization

function sampleSphere(n::Int)
    map(normalize, Slices(randn(3,n),1))
end

a = Spherical(1:7);
@time mean(map(x->abs(harmonic(a/sqrt(dot(a,a)), x))^2,sampleSphere(10000)))*4π

mean(map(x->abs(harmonic(a/sqrt(dot(a,a)), x))^2,sampleSphere(10000)))*4π

# it works perfectly and is about 100x faster than Mathematica
# Everything is done through the generators

## Plot a spherical tensor
n = 120
θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]


a = Spherical(1:7);
##

function complexImage(vals::Array{Complex{Float64}}) 
    max = maximum(abs.(vals))
    map(x -> HSV(angle(x)*360/(2π),(abs(x)/max)^.5,1),vals) 
end

vals = map(((x,y,z) -> harmonic(a/sqrt(dot(a,a)), [x,y,z])), x,y,z);
complexImage(vals)

s = Makie.surface(x, y, z, color = complexImage(vals))

##
