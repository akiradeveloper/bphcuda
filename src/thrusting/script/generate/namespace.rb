# To segregate the namespace of tuple?
# Maybe, should remove

module Thrusting
  private
  def operator_tuple(code="")
"""
namespace thrusting {
#{code}
}
"""
  end
end
