require "pathname"
require "fileutils"
require "kefir"

include FileUtils

# Devices = ["host", "omp", "device"]
Devices = ["device"] # except me need not to run other than GPU. 

# pathname instances
DataDir = Pathname("../data/").realpath
AnalyticDir = Pathname("../analytic_data").realpath

BuildDir = Pathname("../build").realpath
FigureDir = Pathname("#{BuildDir}/figure/plot").realpath
PerformanceDir = Pathname("#{BuildDir}/figure/performance").realpath
TexDir = Pathname("#{BuildDir}/tex").realpath

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
    @dir = dir
    @m = m
  end

  # deprecated
  def get(name)
    @m[name]
  end

  def total_time(backend)
    total = 0
    @m.each do |key, value|
      total += value
    end
    if backend == "host"
      total = total - @m["sort"]
    end
    total
  end

  def get_time(name, backend)
    if name == "total"
      return total_time(backend)
    end
    if name == "sort" and backend == "host"
      return 0
    end
    @m[name]  
  end
end
