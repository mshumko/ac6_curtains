pro rego_skymap

  station_string = 'gill'
  levels = 10
  min_ele = 8
  
  timespan, '2015-09-26/04:30:00', 30 ,/minutes
  thm_load_rego,site=station_string,datatype='rgf'
  get_data, 'clg_rgf_'+station_string, data=asi_d
  thm_load_asi_cal,station_string,asi_cal,/rego
  ;first dimension of x and y are different altitudes. 
  ; if its the same as the white light
  ; 0 - 90 km, 1 - 110 km, 2 - 150 km
  
  
  lon_name='clg_rgf_'+station_string+'_mlon'
  field_index=where(asi_cal.vars.name eq lon_name)
  x=*asi_cal.vars[field_index[0]].dataptr
  x=reform(x[1,*,*])
  lat_name='clg_rgf_'+station_string+'_mlat'
  field_index=where(asi_cal.vars.name eq lat_name)
  y=*asi_cal.vars[field_index[0]].dataptr
  y=reform(y[1,*,*])
  ele_name='clg_rgf_'+station_string+'_elev'
  field_index=where(asi_cal.vars.name eq ele_name)
  elev=*asi_cal.vars[field_index[0]].dataptr
  
  good_elev = where(elev gt min_ele,elev_c)
  
  mlats = y
  mlons = x
  
  e1 = n_elements(mlats[*,0])-1
  e2 = n_elements(mlats[0,*])-1
  
  lat_arr = dblarr(e1,e2)
  lon_arr = lat_arr
  for j=0L,e1-2 do begin
    for k=0L,e2-2 do begin
      lat_arr[j,k] = mean([mlats[j,k],mlats[j+1,k],mlats[j+1,k+1],mlats[j,k+1]], /nan)
      lon_arr[j,k] = mean([mlons[j,k],mlons[j+1,k],mlons[j+1,k+1],mlons[j,k+1]], /nan)
    end
  end
  
  if elev_c gt 0 then begin
    min_lat = min(lat_arr[good_elev],/nan, max=max_lat)
    min_lon = min(lon_arr[good_elev],/nan, max=max_lon)
  endif else begin
    min_lat = min(lat_arr,/nan, max=max_lat)
    min_lon = min(lon_arr,/nan, max=max_lon)
  endelse 
    
  lat_levels = findgen(levels)*(max_lat-min_lat)/levels + min_lat
  lon_levels = findgen(levels)*(max_lon-min_lon)/levels + min_lon

  !p.multi = [0,2,1]
  loadct, 0, /silent
  
  plot, [0,e1],[0,e2], /isotropic, /nodata, $
    xticks = 1, xminor = 1, $
    yticks = 1, yminor = 1
  loadct,0,/silent
  tvscale, alog10(asi_d.y[0,*,*]), /nointerpolation,/overplot   
  loadct, 25,/silent
  contour,lat_arr, /overplot,/follow, levels = lat_levels
  
  

  loadct, 0, /silent
  plot, [0,e1],[0,e2], /isotropic, /nodata, $
    xticks = 1, xminor = 1, $
    yticks = 1, yminor = 1
  loadct,0,/silent
  tvscale, alog10(asi_d.y[0,*,*]), /nointerpolation,/overplot   
  loadct, 25,/silent
  contour,lon_arr, /overplot,/follow, levels = lon_levels
  
  
  
  stop

end