from skyfield.api import EarthSatellite, Topos, load

planets = load('de421.bsp')
earth = planets['earth']
station = earth + Topos(latitude_degrees=45, longitude_degrees=-110, 
                elevation_m=2000.0)
sat = earth + Topos(latitude_degrees=45, longitude_degrees=-200, 
                elevation_m=200E3)

ts = load.timescale()
t = ts.now()

astro = station.at(t).observe(sat)
app = astro.apparent()
alt, az, distance = app.altaz()
print(alt.degrees, az.degrees)