#import GeometryBasics
#import VoronoiCells

# the fraction of area of the disc of radius r centred on u that
# is also covered by one or more of the discs of radius r centred at the other points
function FractionOfContestedArea(u::Point,s::PointSet,r::Real; nd = 100)
    # Place points in a rectangular grid over B(u,r)
    ux = coordinates(u)[1]; uy = coordinates(u)[2];
    uball = Ball(u,r)
    dummypoints = reshape([Point(ux + dx,uy + dy) for dx in range(-r,r,length=nd), dy in range(-r,r,length=nd)],nd*nd)
    filter!(p -> (u ∈ uball),dummypoints)
    inotherballs(p,s) = any(x->p ∈ Ball(x,r),s.items)
    frac = mean(inotherballs(p,s) for p in dummypoints)
    return 1-frac
end




#
# Consider writing a wrapper for VoronoiCells, 
# that spits out centers (as Points) and cells (as Polygons)
#

function getvoronoicells(s::PointSet, W::Box)
    X = s.items
    lo, up = Meshes.coordinates.(extrema(W))
    # need to convert to GeometryBasics
    rect = VoronoiCells.Rectangle(GeometryBasics.Point(lo), GeometryBasics.Point(up))
    pts = [GeometryBasics.Point(Meshes.coordinates(x)) for x in X]
    tess = VoronoiCells.voronoicells(pts, rect)
    cells = tess.Cells
    cells = map(cell->Ngon([Point(v[1],v[2]) for v in cell]...),cells)
    points = tess.Generators
    points = map(pt->Point(pt[1],pt[2]), points)
    return cells, points
end

function polyCircle(p,r; N=20)
    θ = range(0, stop=2*π, length=N)
    points = [Point(r*cos(t)+Meshes.coordinates(p)[1],r*sin(t)+Meshes.coordinates(p)[2]) for t in θ]
    return Ngon(points...)
end

function IntersectConvexPolygons(P1::Ngon,P2::Ngon)
    P1inP2 = filter(x->in(x,P2), P1.vertices)
    P2inP1 = filter(x->in(x,P1), P2.vertices)
    segmentsP1 = [Segment(P1.vertices[i],P1.vertices[i+1]) for i in 1:length(P1.vertices)-1]
    segmentsP2 = [Segment(P2.vertices[i],P2.vertices[i+1]) for i in 1:length(P2.vertices)-1]
    intersectionpoints = filter(!isnothing,[s1 ∩ s2 for s1 in segmentsP1, s2 in segmentsP2])
    points = vcat(P1inP2,P2inP1,intersectionpoints)
    # now sort points by angle (or atan(.,.) )
    pc = Point(sum(Meshes.coordinates, points) / length(points))
    #sort!(points, by= x-> ∠(x-pc,Meshes.Vec(1.,0.)))
    sort!(points, by= x->atan((x-pc)[2],(x-pc)[1])) # atan(y,x)
    return points
    #return Ngon(points...)
end

function BallUnionArea(s::PointSet,r::Real,W::Box; N=12)
    X = s.items
    cells, points = getvoronoicells(s,W)
    # Compute intersection of each ball and its cell
    A = 0
    for k in eachindex(cells)
        B = polyCircle(points[k],r; N=N)
        C = cells[k]
        #D = IntersectConvexPolygons(B,C) # Intersection of the two
        #d = Meshes.coordinates.(D.vertices)
        d = Meshes.coordinates.(IntersectConvexPolygons(B,C))
        #A = A + 0.5*sum((d[i][1]-d[i+1][1])*(d[i][2]+d[i+1][2]) for i in 1:length(d)-1)
        #if nvertices(D)>0
        if length(d)>0
            try 
                #A = A + measure(D)
                A = A + 0.5*sum((d[i][1]-d[i+1][1])*(d[i][2]+d[i+1][2]) for i in 1:length(d)-1)
            catch e
                println(e)
                println(D)
            end
        end
    end
    return A
end



# Alternative

function IntersectConvexPolygons2(P1::Ngon,P2::Ngon)
    P1inP2 = filter(x->in(x,P2), P1.vertices)
    P2inP1 = filter(x->in(x,P1), P2.vertices)
    segmentsP1 = [Segment(P1.vertices[i],P1.vertices[i+1]) for i in 1:length(P1.vertices)-1]
    segmentsP2 = [Segment(P2.vertices[i],P2.vertices[i+1]) for i in 1:length(P2.vertices)-1]
    intersectionpoints = filter(!isnothing,[s1 ∩ s2 for s1 in segmentsP1, s2 in segmentsP2])
    points = vcat(P1inP2,P2inP1,intersectionpoints)
    # now sort points by angle (or atan(.,.) )
    pc = Point(sum(Meshes.coordinates, points) / length(points))
    sort!(points, by= x-> ∠(x-pc,Meshes.Vec(1.,0.)))
    return Ngon(points...)
    #return points
end

function totarea2(tess,r; N=12)
    A = 0
    for k in 1:length(tess.Cells)
        p = Point(tess.Generators[k][1],tess.Generators[k][2])
        B = polyCircle(p,r; N=N) # Ball as polygon
        C = tess.Cells[k]
        Cpts = [Point(c[1],c[2]) for c in C]
        C = Ngon(Cpts...) # voronoi cell as polygon
        D = IntersectConvexPolygons2(B,C) # Intersection of the two
        if nvertices(D)>0
            try 
                A = A + measure(D)
            catch e
                println(e)
                println(D)
            end
        end
    end
    return A
end

function BallUnionArea2(s::PointSet,r::Real,W::Box)
    X = s.items
    lo, up = Meshes.coordinates.(extrema(W))
    # need to convert to GeometryBasics
    rect = VoronoiCells.Rectangle(GeometryBasics.Point(lo), GeometryBasics.Point(up))
    pts = [GeometryBasics.Point(Meshes.coordinates(x)) for x in X]
    # Now compute voronoi cells
    tess = VoronoiCells.voronoicells(pts, rect);
    # And compute total area
    return totarea2(tess,r)
end


