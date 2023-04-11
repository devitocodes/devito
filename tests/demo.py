"""
Profiling tools
---------------
About python profiling and visualization, here's one of my favourite posts:

    https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script

The other tools discussed here

    https://devitocodes.slack.com/archives/C7JMLMSG0/p1678353183480249

Might not work well under Docker. So I'd ignore them for now.


Copy raw data from Docker container to local host
-------------------------------------------------
If you are using Docker like me, to copy the profiling files from the Docker
container to the host you can follow the instructions here:

    https://stackoverflow.com/questions/22049212/docker-copying-files-from-docker-container-to-host

TLDR:

    docker ps
    <copy CONTAINER ID of the container you're running; assume it's called "a4d863150bd4">
    docker cp a4d863150bd4:/app/profiling_demo-startup.data .


Plot generation
---------------
From the link at the top, to plot a nice callgraph from the raw data (e.g.,
`profiling_demo-startup.data`), I first of all install gprof2py:

    sudo apt-get install graphviz  # Compulsory dependence of gprof2dot
    git clone https://github.com/jrfonseca/gprof2dot

Then:

    python gprof2dot/gprof2dot.py -f pstats /app/profiling_demo-startup.data | dot -Tsvg -o profiling_demo-startup.svg

To visualize, I open the .svg file with Chrome.
"""

import cProfile

print("*** DEMO STARTED ***\n")

print("Startup... [profiling ON]")
cProfile.run("""
from devito import *
""", filename='profiling_demo-startup.data')
print()


print("Problem setup...")
configuration['log-level'] = 'DEBUG'

grid = Grid(shape=(16, 16, 16))
t = grid.stepping_dim
x, y, z = grid.dimensions

so = 8

a = Function(name='a', grid=grid, space_order=so)
f = Function(name='f', grid=grid, space_order=so)
e = Function(name='e', grid=grid, space_order=so)
r = Function(name='r', grid=grid, space_order=so)
p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=so)
m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=so)

def g1(field, r, e):
    return (cos(e) * cos(r) * field.dx(x0=x+x.spacing/2) +
            cos(e) * sin(r) * field.dy(x0=y+y.spacing/2) -
            sin(e) * field.dz(x0=z+z.spacing/2))

def g2(field, r, e):
    return - (sin(r) * field.dx(x0=x+x.spacing/2) -
              cos(r) * field.dy(x0=y+y.spacing/2))

def g3(field, r, e):
    return (sin(e) * cos(r) * field.dx(x0=x+x.spacing/2) +
            sin(e) * sin(r) * field.dy(x0=y+y.spacing/2) +
            cos(e) * field.dz(x0=z+z.spacing/2))

def g1_tilde(field, r, e):
    return ((cos(e) * cos(r) * field).dx(x0=x-x.spacing/2) +
            (cos(e) * sin(r) * field).dy(x0=y-y.spacing/2) -
            (sin(e) * field).dz(x0=z-z.spacing/2))

def g2_tilde(field, r, e):
    return - ((sin(r) * field).dx(x0=x-x.spacing/2) -
              (cos(r) * field).dy(x0=y-y.spacing/2))

def g3_tilde(field, r, e):
    return ((sin(e) * cos(r) * field).dx(x0=x-x.spacing/2) +
            (sin(e) * sin(r) * field).dy(x0=y-y.spacing/2) +
            (cos(e) * field).dz(x0=z-z.spacing/2))

update_p = t.spacing**2 * a**2 / f * \
    (g1_tilde(f * g1(p0, r, e), r, e) +
     g2_tilde(f * g2(p0, r, e), r, e) +
     g3_tilde(f * g3(p0, r, e) + f * g3(m0, r, e), r, e)) + \
    (2 - t.spacing * a)

update_m = t.spacing**2 * a**2 / f * \
    (g1_tilde(f * g1(m0, r, e), r, e) +
     g2_tilde(f * g2(m0, r, e), r, e) +
     g3_tilde(f * g3(m0, r, e) + f * g3(p0, r, e), r, e)) + \
    (2 - t.spacing * a)

eqns = [Eq(p0.forward, update_p),
        Eq(m0.forward, update_m)]

print()
print("Generating code version 0... [profiling ON]")
cProfile.run("""
Operator(eqns, subs=grid.spacing_map)
""", filename='profiling_demo-codegen0.data')

print()
print("Generating code version 1... [profiling ON]")
cProfile.run("""
Operator(eqns, subs=grid.spacing_map, opt=('advanced', {'expand': False}))
""", filename='profiling_demo-codegen1.data')

print()
print("*** DEMO OVER! ***\n")

print("If you are inside a Docker container and need to copy the .data files over to the host, from the host run:")
import socket
hostname = socket.gethostname()
print("docker cp %s:/app/profiling_demo-startup.data ." % hostname)
print("docker cp %s:/app/profiling_demo-codegen0.data ." % hostname)
print("docker cp %s:/app/profiling_demo-codegen1.data ." % hostname)
print("(Replace `%s` with actual Container ID if not matching)")

print()
print("If you now want to generate the SVG plots, run:")
print("""python3 gprof2dot.py -f pstats profiling_demo-startup.data | dot -Tsvg -o profiling_demo-startup.svg
python3 gprof2dot.py -f pstats profiling_demo-codegen0.data | dot -Tsvg -o profiling_demo-codegen0.svg
python3 gprof2dot.py -f pstats profiling_demo-codegen1.data | dot -Tsvg -o profiling_demo-codegen1.svg""")
