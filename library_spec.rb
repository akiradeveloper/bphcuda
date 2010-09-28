# NOTE
# NEVER write complicated test that opts to be debt.

require File.dirname(__FILE__) + "/" + "library"

describe "ruby-library-mph" do

  describe "count up the particles in side" do
    it "works fine" do
      aparticle = [1.2, 1.2, 1.2]
      particles = Array.new(4){aparticle}
      origin = [0,0,0]
      cell_spaces = [1,1,1]
      cell_dims = [2,2,2]
      ary = Array.new(2*2*2){0}
      count_up_particles!(particles, origin, cell_spaces, cell_dims, ary)
      ary.should == [0,0,0,0,0,0,0,4]
    end
  end

  describe "cell index" do
    it "detects the cell index in 3D" do
      origin = [0.0, 0.0, 0.0]
      cell_spaces = [1.0, 1.0, 1.0]
      p = [1.2, 1.2, 1.2]
      detect_index(origin, cell_spaces, p).should == [1,1,1]
    end
  end

  describe "convert from 3D to 1D array" do
    it "works fine" do
      cell_dims = [2,2,2]
      conv2aryind(cell_dims, [1,1,1]).should == 7
    end
  end
 
  describe "add!" do
    it "works fine with me" do
      dest = Array.new(3){0}
      add!([1,2,3], [4,5,6], dest)
      dest.should == [5,7,9]
    end
  end  

  describe "prefix sum exclusive" do
    it "works as we desire" do
      src = [1,2,3,4,5]
      dest = Array.new(4){0}
      prefix_sum_exclusive!(src, 0, 3, dest)
      dest.should == [0, 1, 3, 6]
    end
  end
  
  describe "radix sort" do
    it "works fine with small case" do
      keys = [3,2,1]
      values = ['a', 'b', 'c']
      radix_sort_by_key!(keys, 0, 1, values)
      keys.should == [2,3,1]
      values.should == ['b', 'a', 'c']
    end
    it "works fine with complicated case" do
      keys = [3,2,1]
      values = ['a', 'b', 'c']
      radix_sort_by_key!(keys, 1, 2, values)
      keys.should == [3,1,2]
      values.should == ['a', 'c', 'b']
    end
  end
end
