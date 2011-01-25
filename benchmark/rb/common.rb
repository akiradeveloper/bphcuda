require "pathname"

require "kefir"

Devices = ["host", "omp", "device"]

# pathname instances
DataDir = Pathname("../data/").realpath
AnalyticDir = Pathname("../analytic_data").realpath
FigureDir = Pathname("../figure").realpath

def devicename(backend)
  case backend
  when "host"
    "CPU"
  when "device"
    "GPU"
  when "omp"
    "OpenMP"
  else
    raise
  end
end

class TimeData

  def initialize(dir)
    f = open(dir + "/" + "time.dat")
    content = f.read.split("\n")
    m = {}
    content.each do |x|
      t = x.split(":")
      key = t[0]
      value = t[1].to_f
      m[key] = value
    end
    @m = m
  end

  def get(name)
    @m[name]
  end

  def total_time(backend)
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
end
