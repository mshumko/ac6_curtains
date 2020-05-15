;plot a lat and lon grid on the ASIs




pro asi_grid

  del_data,'*'

  out_dir = 'C:\Users\krmurph1\Physics\output\AuroraEpoch\'
  out_dir = 'D:\output\ASI_Grid\'

  station  = 'inuv'
  asi_date = '2013-03-30/06:00:00'
  duration = 1

  
  
  mag = 0
  ps = 0
  outpng = 0
  
  
  if mag eq 1 then begin
    latp = 9
    lonp = 8
  endif else begin
    latp = 7
    lonp = 6
  endelse
  
  asi_ct = 9
  
  lat_step = 2
  lon_step = 5
  
  min_elevation = 6
  
  timespan,asi_date,duration,/minutes

  thm_load_asi,site=strlowcase(station), datatype = asf
  thm_load_asi_cal,strlowcase(station),asi_cal
  get_data, 'thg_asf_'+strlowcase(station), data = d, dtype = data_type, index = data_index
  ;get the asi 256x256 data 
  ;stop
  ;get elevation data here for each pixel
  ele_name='thg_asf_'+strlowcase(station)+'_elev'
  field_index=where(asi_cal.vars.name eq ele_name)
  elev_asi =reform(*asi_cal.vars[field_index[0]].dataptr)
  ; get rid of pixel with bad elevation
  ; do initial analysis on entire frame 
  bad_ele = where(elev_asi le min_elevation or finite(elev_asi) ne 1,c_bad)
  good_ele = where(elev_asi gt min_elevation and finite(elev_asi) eq 1, c_good)
  inf_ele = where(finite(elev_asi) ne 1,c_inf)
  
  ; get  latitude and longitude here
  ; variable #7 is latitude, variable #6 is longitude
  ; variable #9 is magnetic latitude, variable #8 is magnetic longitude
  ; x-dimension is different altitude with 0-90 km, 1-110 km, and 2-150 km
  ; magnetic
  mlats=reform((*asi_cal.vars[latp].dataptr)[1,*,*])   
  mlons=reform((*asi_cal.vars[lonp].dataptr)[1,*,*])
  ; create latitude array
  lat_hist = dblarr(256,256)
  lon_hist = lat_hist
  for j=0L,n_elements(mlats[*,0])-2 do begin      ;256 elements in j  
    for k=0L,n_elements(mlats[*,0])-2 do begin      ;256 elements in k  
      lat_hist[j,k] = mean([mlats[j,k],mlats[j+1,k],mlats[j+1,k+1],mlats[j,k+1]], /nan)
      lon_hist[j,k] = mean([mlons[j,k],mlons[j+1,k],mlons[j+1,k+1],mlons[j,k+1]], /nan)
    endfor
  endfor
  
  if ps eq 1 then begin
    set_plot, 'ps'
    out_file = out_dir+strupcase(station)+'.ps'
    device, xsize=11, ysize=4.25, bits_per_pixel=8, /color, /inches, $
      filename = out_file, xoffset=0, yoffset=0 
    char_plot = 1.25
  endif else begin
    window, xsize = 1550, ysize = 550
    char_plot = 1.5  
  endelse
  !p.multi   = [0,3,1]
  !x.margin  = [5,0]
  !y.margin  = [5,0]
  !x.omargin = [2,18]
  !y.omargin = [0,0]
  !x.style = 1
  !y.style = 1
  
  bright_pt = total(total(d.y,2),2)
  max_br    = max(bright_pt)
  p_pos = !C
    
  plot_asi = double(reform(d.y[p_pos,*,*]))
  ;plot_asi[bad_ele] = !values.f_nan
  
  min_asi = min(plot_asi, /nan)
  min_asi = (min_asi/10^(long(alog10(min_asi))))*10^(long(alog10(min_asi)))
  max_asi = 10^(long(alog10(min_asi))+1)

  ;plot lat grid
  loadct, 0
  plot, [0,255],[0,255], /isotropic, /nodata, xticks=1, xminor=1, yticks=1, yminor=1, ytickf='no_ticks', xtickf='no_ticks', charsize = 2., charthick = 1.5
  axis, yaxis = 0, ytitle = 'CGM latitude', xticks=1, xminor=1, yticks=1, yminor=1, ytickf='no_ticks', xtickf='no_ticks',charsize = 2
  top_left = convert_coord(!x.crange[0],!y.crange[1], /data, /to_normal)
  xyouts, top_left[0], top_left[1]+0.01, station,/normal, charsize = 1.5
  loadct, asi_ct
  tvscale, alog10(plot_asi), minvalue = alog10(min_asi), maxvalue = alog10(max_asi), /overplot,/nointerpolation

  
  
  start_lat = long(min(lat_hist[good_ele],/nan, max=max_lat)/lat_step)*lat_step
  end_lat = max_lat
  
  npts = round((end_lat-start_lat)/lat_step)
  
  plot_lat = indgen(npts)*lat_step+start_lat
  
  mid_pt = round(lat_hist[127,127])
  
  min_diff = min(abs(plot_lat-mid_pt),/nan)
  
  if min_diff gt 0 then plot_lat = [plot_lat,mid_pt]
  
  lat_sort = sort(plot_lat)
  plot_lat = plot_lat[lat_sort]
  
  leg_pos = indgen(n_elements(plot_lat))*((256-32*2)/n_elements(plot_lat))+32
  
  
  loadct, 10
  pc = 191
  for j = 0L, n_elements(plot_lat)-1 do begin
    start_lat  = plot_lat[j]
    print, start_lat
    plot_point = 0
    leg_plot   = 0
    for i=0L, n_elements(lat_hist[*,0])-1 do begin
      lat_arr = reform(lat_hist[i,*])
      min_diff = min(abs(lat_arr-start_lat),/nan)
      min_pos = !C
      
      if min_diff gt 0.1 then continue
      if elev_asi[i,min_pos] lt min_elevation or finite(elev_asi[i,min_pos]) eq 0 then continue
      
      if plot_point eq 0 then plots, i,min_pos, psym = 3, color = pc
      plots, i,min_pos, color = pc, /continue, thick = 1.5    
      plot_point++
      if i eq leg_pos[j] then begin
        xyouts, i+3,min_pos-10, strtrim(long(plot_lat[j]),2)+'!Uo!N', /data, color = pc, charsize = char_plot, charthick = char_plot
        leg_plot = 1
        plot_point = 0
        if ps eq 1 then i = i+25 else i = i+20
      endif else if i gt leg_pos[j] and leg_plot eq 0 and plot_point gt 10 then begin
        xyouts, i+3,min_pos-10, strtrim(long(plot_lat[j]),2)+'!Uo!N', /data, color = pc, charsize = char_plot, charthick = char_plot
        leg_plot = 1
        plot_point = 0
        if ps eq 1 then i = i+25 else i = i+20
      endif
    endfor
  endfor
  
  
  ;plot longitude grid
  loadct, 0
  plot, [0,255],[0,255], /isotropic, /nodata, xticks=1, xminor=1, yticks=1, yminor=1, ytickf='no_ticks', xtickf='no_ticks', charsize = 2., charthick = 1.5
  axis, xaxis=0, xtitle = 'CGM longitude', xticks=1, xminor=1, yticks=1, yminor=1, ytickf='no_ticks', xtickf='no_ticks',charsize = 2
  top_left = convert_coord(!x.crange[0],!y.crange[1], /data, /to_normal)
  loadct, asi_ct
  tvscale, alog10(plot_asi), minvalue = alog10(min_asi), maxvalue = alog10(max_asi), /overplot,/nointerpolation

  start_lon = long(min(lon_hist[good_ele],/nan, max=max_lon)/lon_step)*lon_step
  end_lon = max_lon
  
  npts = round((end_lon-start_lon)/lon_step)
  
  plot_lon = indgen(npts)*lon_step+start_lon
  
  mid_pt = round(lon_hist[127,127])
  
  min_diff = min(abs(plot_lon-mid_pt),/nan)
  
  if min_diff gt 0 then plot_lon = [plot_lon,mid_pt]
  
  lon_sort = sort(plot_lon)
  plot_lon = plot_lon[lon_sort]
  
  leg_pos = indgen(n_elements(plot_lon))*((256-32*2)/n_elements(plot_lon))+32
  
  
  loadct, 10
  pc = 191
  for j = 0L, n_elements(plot_lon)-1 do begin
    start_lon  = plot_lon[j]
    print, start_lon
    plot_point = 0
    leg_plot   = 0
    for i=0L, n_elements(lon_hist[0,*])-1 do begin
      lon_arr = reform(lon_hist[*,i])
      min_diff = min(abs(lon_arr-start_lon),/nan)
      min_pos = !C
      
      if min_diff gt 0.1 then continue
      if elev_asi[min_pos,i] lt min_elevation or finite(elev_asi[min_pos,i]) eq 0 then continue
      
      if plot_point eq 0 then plots, i,min_pos, psym = 3, color = pc
      plots, min_pos, i, color = pc, /continue, thick = 1.5    
      plot_point++
      if i eq leg_pos[j] then begin
        xyouts, min_pos-7,i+3, strtrim(long(plot_lon[j]),2)+'!Uo!N', /data, color = pc, charsize = char_plot, charthick = char_plot, alignment = 0.5
        leg_plot = 1
        plot_point = 0
        i = i+20
      endif else if i gt leg_pos[j] and leg_plot eq 0 and plot_point gt 10 then begin
        xyouts, min_pos-7, i+3, strtrim(long(plot_lon[j]),2)+'!Uo!N', /data, color = pc, charsize = char_plot, charthick = char_plot, alignment = 0.5
        leg_plot = 1
        plot_point = 0
        i = i+20
      endif
    endfor
  endfor
  
  plot, [0,255],[0,255], /isotropic, /nodata, xticks=1, xminor=1, yticks=1, yminor=1, ytickf='no_ticks', xtickf='no_ticks', charsize = 2, charthick = 1.5  
  loadct, asi_ct,/silent
  tvscale, alog10(plot_asi), minvalue = alog10(min_asi), maxvalue = alog10(max_asi), /overplot,/nointerpolation
  loadct, 10
  pc = 191
  c_lev = findgen(10)*10
  c_lab = indgen(10)
  c_lab[*] = 1
  c_col = c_lab
  c_col[*] = pc
  contour, elev_asi, levels =c_lev, c_labels=c_lab, c_colors=c_col ,/overplot, c_charsize = char_plot, c_charthick = char_plot, c_thick = 1.5
  axis, xaxis = 0, xtitle = 'Elevation', xticks=1, xminor=1, yticks=1, yminor=1, ytickf='no_ticks', xtickf='no_ticks',charsize = 2
  if ps eq 1 then begin
    device,/close
    set_plot, 'win'
  endif

  if ps ne 1 and outpng eq 1 then begin
    makepng, out_dir+station
  endif

  

end