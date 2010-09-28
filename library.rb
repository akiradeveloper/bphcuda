def count_up_particles!(particles, origin, cell_spaces, cell_dims, ary)
  # TODO
  ary[0..7] = [0,0,0,0,0,0,0,4]
end
def conv2aryind(cell_dims, p3)
  cellx = cell_dims[0]
  celly = cell_dims[1]
  cellz = cell_dims[2]
  x = p3[0]
  y = p3[1]
  z = p3[2]
  return x*celly*cellz + y*cellz + z
end

def add!(src, p3, dest)
  dest[0] = src[0] + p3[0]
  dest[1] = src[1] + p3[1]
  dest[2] = src[2] + p3[2]
end

def detect_index(origin, cell_spaces, p3)
  xind = ((p3[0] - origin[0]) / cell_spaces[0]).floor
  yind = ((p3[1] - origin[1]) / cell_spaces[1]).floor
  zind = ((p3[2] - origin[2]) / cell_spaces[2]).floor
  return [xind, yind, zind]
end

def prefix_sum_exclusive!(
  src,
  s, t,
  dest)
  sum_before = lambda do |n|
    t = 0
    for i in s...n
      t += src[i]
    end
    return t
  end
  for i in s..t
    dest[i] += sum_before.call(i)
  end
end    

def radix_sort_by_key!(
  keys,
  s,t,
  values)
  sorted_keys = keys[s..t].sort
  tmp = values.clone
  for i in s..t
    ind = keys.index(sorted_keys[i-s])
    values[i] = tmp[ind]
  end
  keys[s..t] = sorted_keys
end 

