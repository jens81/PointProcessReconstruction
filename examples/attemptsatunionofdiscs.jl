
##########################################
####### Area of union of circles

using VoronoiCells
using GeometryBasics

function CircleShape(p::GeometryBasics.Point2,r)
    θ = range(0, stop=2*π, length=50)
    x = r.*cos.(θ) .+ p[1]
    y = r.*sin.(θ) .+ p[2]
    return Shape(x,y)
end

rect = Rectangle(GeometryBasics.Point2(0, 0), GeometryBasics.Point2(1, 1))
points = [GeometryBasics.Point2(rand(), rand()) for _ in 1:100]
tess = voronoicells(points, rect);
#tess.Cells[1]
scatter(points, markersize = 3, label = "generators", xlims=(0,1),ylims=(0,1),aspectratio=1)
annotate!([(points[n][1] + 0.02, points[n][2] + 0.03, Plots.text(n,5)) for n in 1:length(points)])
plot!(tess, legend = :topleft)
plot!([CircleShape(p,0.05) for p in points], opacity=0.2, color=:gray, aspectratio=1, label=false)




scatter(points, markersize = 3, label = "generators", xlims=(0.3,.8),ylims=(0.3,0.8),aspectratio=1)
annotate!([(points[n][1] + 0.02, points[n][2] + 0.03, Plots.text(n,5)) for n in 1:length(points)])
plot!(tess, legend = :bottomleft)
plot!([CircleShape(p,0.05) for p in points], opacity=0.2, color=:gray, aspectratio=1, label=false)



plot([CircleShape(p,0.07) for p in points], opacity=1, color=:gray, aspectratio=1, label=false, linecolor=:gray)
plot!(tess, legend = :bottomleft, aspectratio=1, linecolor=:black)


function findintersectionpoints(v1,v2,B)
    r = B.r
    p1 = v1 - B.center
    p2 = v2 - B.center
    D = p1[1]*p2[2] - p1[2]*p2[1]
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    dr = sqrt(dx^2+dy^2)
    Δ = r^2*dr^2 - D^2
    sgn(x) = (x<0) ? -1 : 1
    if Δ<=0
        return nothing, nothing
    else
        q1 = GeometryBasics.Point( (D*dy + sgn(dy)*dx*sqrt(Δ))/(dr^2), (-D*dx + sgn(dy)*dy*sqrt(Δ))/(dr^2) )
        q2 = GeometryBasics.Point( (D*dy - sgn(dy)*dx*sqrt(Δ))/(dr^2), (-D*dx - sgn(dy)*dy*sqrt(Δ))/(dr^2) )
        return q1+B.center, q2+B.center
    end
end

function circlesegmentarea(q1,q2,B)
    if isnothing(q1) || isnothing(q2)
        return 0.0
    else 
        r = B.r
        dx = q2[1] - q1[1]
        dy = q2[2] - q1[2]
        dr = sqrt(dx^2+dy^2)
        θ = 2*asin(dr/(2*r))
        A = (θ - sin(θ))*r^2/2
        return A
    end
end

circlesegmentarea(a,b,B)
π*(B.r)^2

function overlaparea(q1,q2,p,B)
    if isnothing(q1) || isnothing(q2) || isnothing(p)
        return 0.0
    else 
        A1 = abs(GeometryBasics.area([q1,q2,p]))
        A2 = circlesegmentarea(q1,q2,B)
        return A1 + A2
    end
end

l1 = Segment((a1[1],a1[2]),(b1[1],b1[2]))
l2 = Segment((a2[1],a2[2]),(b2[1],b2[2]))
p = l1 ∩ l2
p = GeometryBasics.Point(Meshes.coordinates(p))
overlaparea(a,b,p,B)



function closest(points, p)
    d(q,p) = sqrt((p[1]-q[1])^2+(p[2]-q[2])^2)
    dists = [d(q,p) for q in points]
    (mindist, i) = findmin(dists)
    return points[i]
end

function segmentintersectionpoint(pts1,pts2)
    l1 = Segment( (pts1[1][1],pts1[1][2]),(pts1[2][1],pts1[2][2]) )
    l2 = Segment( (pts2[1][1],pts2[1][2]),(pts2[2][1],pts2[2][2]) )
    p = l1 ∩ l2
    if isnothing(p)
        return nothing
    else
        return GeometryBasics.Point(Meshes.coordinates(p))
    end
end


l1 = Segment( (0,0), (1,1) )
l2 = Segment( (1,0), (1,1) )
p = l1 ∩ l2
isnothing(p)
isnothing(GeometryBasics.Point(nothing))

V = [C...,C[1]]

function AreaInCell(B,cell)
    k = length(cell)
    V = [cell...,cell[1]]
    SegmentAreas = zeros(k)
    OverlapAreas = zeros(k,k)
    q1 = Array{Union{Nothing, GeometryBasics.Point2}}(nothing, k)
    q2 = Array{Union{Nothing, GeometryBasics.Point2}}(nothing, k)
    for i in 1:k
        #println(i)
        q1[i],q2[i] = findintersectionpoints(V[i],V[i+1],B)
        SegmentAreas[i] = circlesegmentarea(q1[i],q2[i],B)
        for j in 1:i
            if j==i || SegmentAreas[j] == 0 || SegmentAreas[i] == 0
                OverlapAreas[i,j] = 0
            else 
                p = segmentintersectionpoint([q1[i],q2[i]],[q1[j],q2[j]])
                if isnothing(p)
                    OverlapAreas[i,j] = 0
                else
                    # compute overlap between i and j
                    a = closest([q1[i],q2[i]],p)
                    b = closest([q1[j],q2[j]],p)
                    #println(a,b,p)
                    OverlapAreas[i,j] = overlaparea(a,b,p,B)
                end
            end
        end
    end
    #println(SegmentAreas)
    #println(OverlapAreas)
    #println(π*B.r^2)
    #println(sum(SegmentAreas))
    #println(sum(OverlapAreas))
    return π*B.r^2 - sum(SegmentAreas) + sum(OverlapAreas)
end

AreaInCell(B,C)
AreaInCell(B,C)/(π*B.r^2)


# Try different cell
c = 61
C = tess.Cells[c]
B = GeometryBasics.Circle(tess.Generators[c],0.05)
AreaInCell(B,C)
AreaInCell(B,C)/(π*B.r^2)

function totarea(tess,r)
    A = 0
    for c in 1:length(tess.Cells)
        C = tess.Cells[c]
        B = GeometryBasics.Circle(tess.Generators[c],r)
        A = A + AreaInCell(B,C)
    end
    return A
end

totarea(tess,0.1)