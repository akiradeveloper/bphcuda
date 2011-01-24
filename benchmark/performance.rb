require "kefir"

def name(n, m, backend) 
  "data/shocktube_x_n#{n}_m#{m}_s2_#{backend}"
end

def time(dir)
  f = open(dir + "/" + "time.dat")
  content = f.read.split("\n")
  m = {}
  content.each do |x|
    t = x.split(":")
    key = t[0]
    value = t[1].to_f
    m[key] = value
  end
  m
end

def total_time(map, backend)
  total = 0
  map.each do |key, value|
    total += value
  end
  if backend == "host"
    total = total - map["sort"]
  end
  print backend, ":", total, "\n"
  total
end

if __FILE__ == $0
  n = 500
  ms = [1,10,100,1000]
  Kefir.open() do |gp|
    gp.set("terminal", "jpeg")
    x = "shocktube_x"
    gp.set("output", "#{x}.jpeg".embed)
    gp.set("title", x.embed)
    xs = ms
    devices = ["host", "device", "omp"]
    gp.plot do |p| 
      devices.each do |backend|
        p backend
        ys = ms.map { |m| total_time(time(name(n, m, backend)), backend) }
          p << Kefir.eval([xs, ys]) do |d|
            d << "title #{backend.embed}"
            d << "with lines"
          end
      end
    end
  end
end
