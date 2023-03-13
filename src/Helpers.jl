function partitions(W::Box,dim::Tuple{Int64,Int64})
    lo, up = coordinates.(extrema(W))
    dx = (up[1]-lo[1])/dim[1]
    dy = (up[2]-lo[2])/dim[2]
    boxmat = [Box((lo[1]+k*dx,lo[2]+j*dy),(lo[1]+(k+1)*dx,lo[2]+(j+1)*dy)) for k=0:(dim[1]-1), j=0:(dim[2]-1)]
    return reshape(boxmat,dim[1]*dim[2])
end

partitions(W::Box,n::Int) = partitions(W,(n,n))

#function partitions(W::Box,n::Int)
#    lo, up = coordinates.(extrema(W))
#    dx = (up[1]-lo[1])/n
#    dy = (up[2]-lo[2])/n
#    boxmat = [Box((lo[1]+k*dx,lo[2]+j*dy),(lo[1]+(k+1)*dx,lo[2]+(j+1)*dy)) for k=0:(n-1), j=0:(n-1)]
#    return reshape(boxmat,n*n)
#end

function Box2Shape(b::Box)
    lo, up = coordinates.(extrema(b))
    return Shape([lo[1],up[1],up[1],lo[1]],[lo[2],lo[2],up[2],up[2]])
end

# Hack to define empty point sets
function EmptyPointSet()
    s = PointSet([0.,0.])
    deleteat!(s.items,1)
    return s
end

function enlarge(b::Box, r::Real)
    lo, up = coordinates.(extrema(b))
    lo = Tuple(lo .- r)
    up = Tuple(up .+ r)
    return Box(lo,up)
end

function reduce(b::Box, r::Real)
    lo, up = coordinates.(extrema(b))
    lo = Tuple(lo .+ r)
    up = Tuple(up .- r)
    return Box(lo,up)
end

function quadint(f::Function, W::Box; nd=50)
    boxes = partitions(W,nd)
    d = PointSet([centroid(box) for box in boxes])
    a = Meshes.measure(W)/(nd*nd)
    return sum(f(u)*a for u in d.items)
end