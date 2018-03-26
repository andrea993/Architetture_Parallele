#include <iostream>
#include <CL/cl.hpp>
#include <vector>
#include <string>

int main()
{

	// PLATFORMS
	std::vector<cl::Platform> all_platforms;

	cl::Platform::get(&all_platforms);

	if(all_platforms.size()==0)
	{
		std::cerr<<"No platforms found. Install OpenCL"<<std::endl;
		return -1;
	}

	std::cout <<"Platform list:"<<std::endl;
	for (int i=0; i<all_platforms.size();i++)
		std::cout << all_platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

	cl::Platform default_platform=all_platforms[0];
	std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

	// DEVICE
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

	std::cout <<"Device list:"<<std::endl;
	for (int i=0; i<all_devices.size();i++)
		std::cout << all_devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;


	if(all_devices.size()==0)
	{
		std::cerr<<" No devices found. Check OpenCL installation!\n";
		return -1;
	}
	cl::Device default_device=all_devices[0];
	std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<std::endl;


	cl::Context context({default_device});
	cl::Program::Sources sources;

	std::string kernel_code(
			"	void kernel simple_add(global const int* A, global const int* B, global int* C)	"
			"	{																													"
			"		C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];								"
			"	}																													");

	sources.push_back({kernel_code.c_str(),kernel_code.length()});


	cl::Program program(context,sources);
	if(program.build({default_device})!=CL_SUCCESS){
		std::cerr<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
		return -1;
	}

	cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
	cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
	cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

	//create queue to which we will push commands for the device.
	cl::CommandQueue queue(context,default_device);

	//write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
	queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);

	auto simple_add=cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "simple_add");
	cl::EnqueueArgs eargs(queue,cl::NullRange,cl::NDRange(10),cl::NullRange);

	simple_add(eargs, buffer_A,buffer_B,buffer_C).wait();


	int C[10];
	//read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

	std::cout<<" result: \n";
	for(int i=0;i<10;i++){
		std::cout<<C[i]<<" ";
	}

	return 0;

}
