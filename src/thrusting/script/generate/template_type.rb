module Thrusting 
  private
  def typename(range)
    range.map { |i| "typename X#{i}" }.join ", "
  end
  def type(range)
    range.map { |i| "X#{i}" }.join ", "
  end
  def arg(range)
    range.map { |i| "x#{i}" }.join ", "
  end
  def explicit_type_arg(type, range)
    range.map { |i| "#{type} x#{i}" }.join ", "
  end
  def type_arg(range)
    range.map { |i| "X#{i} x#{i}" }.join ", "
  end
end
