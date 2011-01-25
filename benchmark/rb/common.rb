require "pathname"
require "fileutils"
require "kefir"

include FileUtils

Devices = ["host", "omp", "device"]

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
    @m = m
  end

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
end
