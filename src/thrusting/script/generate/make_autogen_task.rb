module Thrusting
  private
  def chain_task(taskname, dir)
    task taskname do
      Dir.chdir(dir) do 
        sh "rake #{taskname}"
      end
    end
  end
  def make_autogen_task(dir)
    chain_task("build", dir)
    chain_task("clobber", dir)
  end
end
