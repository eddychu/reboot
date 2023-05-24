---
title: "Build a Rasterizer With Webgpu 01"
date: 2023-05-24T08:45:31+08:00
draft: false
---

Now since WebGPU is offcially rolled out in Chrome browser, let's try it out by building a rasterizer using this new API.

## Basic Setup

First, we need to check everything is ready for WebGPU. We can do this by checking if the `navigator.gpu` is available.

```ts
if (!navigator.gpu) {
  console.error('WebGPU is not supported');
  return;
}
```
If you have error in your console, that means you need to download the latest version of Chrome. If you are on Linux, As of now, you have to download the dev version of Chrome and enable the WebGPU flag, you probably also need to enable the Vulkan flag and install the Vulkan SDK.

Next, we need to create a GPU device.

```ts
// first we need to request an adapter
// an adapter is a physical device that can be used to render graphics, namely a GPU
const adapter = await navigator.gpu.requestAdapter();

// then we need to request a device
// think of a device as a logical handle to a physical device
const device = await adapter.requestDevice();

```

Now we have a device which we can use to create other WebGPU objects.

As we are just starting out,  we will not bother ourselves with the canvas API, instead we will get a basic compute shader running and send data to it.

## Compute Shader

A compute shader is a program that runs on the GPU. It is used to perform general purpose computation on the GPU. For people we are not familiar with the graphics pipeline, it is a good place to start. As you are free from the vertex and fragment shader complexity, instead you can treat the GPU as a parallel processor, and you feed it with data and get the result back.


### Create a Compute Pipeline

In order to use compute shader, we need to create a compute pipeline. A compute pipeline is a collection of GPU state that controls how the compute shader is executed. 

Other than the compute shader, you also need to specify what data you want to pass to the shader. This is done by creating a bind group layout and bind group.

You can think of a bind group layout as a blueprint of the data you want to pass to the shader. It specifies the type of the data and the binding index. The bind group is the actual data you want to pass to the shader. It is created based on the bind group layout.

the shader content :

```wgsl
  struct ColorBuffer{
    values: array<u32>
  };

  // this is the final output buffer from our shader.
  @group(0) @binding(0) var<storage, read_write> color_buffer: ColorBuffer;

  // we are using workgroup_size(1) here 
  // this is stupid, but it is easy to see how the global_invocation_id works.
  @compute @workgroup_size(1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // multiply by 3 because we have 3 channels
    let index = global_id.x * 3;
    color_buffer.values[index] = index;
    color_buffer.values[index + 1] = index + 1;
    color_buffer.values[index + 2] = index + 2;
  }
```
we create the compute pipeline like this:

```ts
  const width = 100;
  const height = 100;
  const channels = 3;
  // each pixel has 3 channels, r, g, b and each channel is a 32 bit unsigned integer
  // we are wasting a lot of space here, but it saves
  // us from doing some bit shifting in the shader.
  const colorBufferSize = width * height * Uint32Array.BYTES_PER_ELEMENT * channels;
  const colorBuffer = device.createBuffer({
    size: colorBufferSize,

    // we need to specify the usage of the buffer
    // here we are using the buffer as a storage buffer and a copy source
    // we specify the copy source usage because we want to copy the data from the buffer to the CPU which we will do later.
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // create bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
    ]
  });

  // create bind group
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: colorBuffer
        }
      },
    ]
  });

  // create shader module
  const computeShaderModule = device.createShaderModule({
    code: computeShaderCode
  });

  // create compute pipeline
  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: computeShaderModule, entryPoint: "main" }
  });

```

With those prepartions in place we are finally ready to do some computation on the GPU.

### Run the Compute Shader

```ts
  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, bindGroup);
  computePass.dispatchWorkgroups(width * height);
  computePass.end();

  // submit the command encoder to the GPU
  device.queue.submit([commandEncoder.finish()]);
```


### Copy the Data Back to CPU

By now the gpu has finished the computation for us, in this case, the computation is simple, it just set the buffer value to the buffer index.

Now we need to copy the data back to the CPU so we can see the result.

```ts
  // create a staging buffer
  const stagingBuffer = device.createBuffer({
    size: colorBufferSize,
    // here we use the buffer as a copy destination and a map read
    // map read means we can map the buffer to the CPU
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // copy the data from the color buffer to the staging buffer
  commandEncoder.copyBufferToBuffer(colorBuffer, 0, stagingBuffer, 0, colorBufferSize);

  // submit the command encoder to the GPU
  device.queue.submit([commandEncoder.finish()]);

  // map the staging buffer to the CPU
  await stagingBuffer.mapAsync(GPUMapMode.READ);

  // get the array buffer from the staging buffer
  const arrayBuffer = stagingBuffer.getMappedRange();

  const data = new Uint32Array(arrayBuffer.slice(0));

  arrayBuffer.unmap();
  // log the data
  console.log(data);
```
In console, you will see an array of number from 0 to 9999, which is the index of the buffer.


## Graphics Pipeline

So many lines of code to just get a bunch of useless numbers, that's not very interesting. Let's try to do something more interesting, like rendering our previous generated data to the screen.


### Create a Graphics Pipeline

In order to render something to the screen, we need to create a graphics pipeline. A graphics pipeline is a collection of GPU state that controls how the vertex and fragment shader is executed.

A thorough explanation of the graphics pipeline is beyond the scope of this article, please consult other resources if you are interested.

In our case, we are not going to do a proper vertex transformation and fragment shading because we are not providing any vertex data, we are just going to render a color buffer (previously generated) to the screen.

let's set up our vertex and fragment shader first

```wgsl
// this is the data we are going to pass to the shader
// it has the same structure as the data we generated in the compute shader
struct ColorData {
    values: array<u32>
};

struct Uniform {
    screenWidth: f32,
    screenHeight: f32
};

@group(0) @binding(0) var<uniform> uniforms: Uniform;
@group(0) @binding(1) var<storage> color_data: ColorData;

struct VertexOutput {
    @builtin(position) position: vec4<f32>
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    return output;
}


@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = floor(pos.x);
    let y = floor(pos.y);
    let index = u32(x + y * uniforms.screenWidth) * 3;

    let combinations = uniforms.screenWidth * uniforms.screenHeight;

    // this is u32
    let r = f32(color_data.values[index + 0]) / pixels;
    let g = f32(color_data.values[index + 1]) / pixels;
    let b = f32(color_data.values[index + 2]) / pixels;

    return vec4<f32>(r, g, b, 1.0);
}

```
the vertex shader and fragment shader is simple, we bind the color buffer to the fragment shader and use the buffer index to get the data value.

the graphics pipeline creation follows the same routine as with the compute pipeline. The only difference is that we need to create a canvas and a canvas context to render the result to the screen.

```ts
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  // set the width and height of the canvas to our color buffer size.
  canvas.width = width;
  canvas.height = height;

  // create a canvas context
  const context = canvas.getContext("webgpu");

  const format = "bgra8unorm";

  context.configure({
    device,
    format,
  });

 const graphicsShader = device.createShaderModule({
    code: graphicsShaderSource
  });


  const 

  // other than the color buffer we created before, 
  // we binded a uniform buffer to the graphics pipeline
  // the uniform buffer contains the screen width and height
  // we need this to calculate the index of the color buffer
  const graphicsBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform"
        }
      },
      {
        binding: 1,// the color buffer
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "read-only-storage"
        }
      }
    ]
  });


  const graphicsPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [graphicsBindGroupLayout]
    }),
    vertex: {
      module: graphicsShader,
      entryPoint: "vs_main",
    },
    fragment: {
      module: graphicsShader,
      entryPoint: "fs_main",
      targets: [
        {
          format: swapChainFormat,
        }
      ]
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // uniform buffer contains the screen width and height. 
  const uniformBufferSize = 4 * 2;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([width, height])
  );

  const graphicsBindGroup = device.createBindGroup({
    layout: graphicsBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        }
      },
      {
        binding: 1,
        resource: {
          buffer: colorBuffer,
        }
      }
    ]
  });

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: currentTexture.createView(),
        clearValue: [1.0, 0.0, 0.0, 1.0],
        loadOp: "clear",
        storeOp: "store"
      }
    ]
  };
 const renderCommandEncoder = device.createCommandEncoder();
  const renderPass = renderCommandEncoder.beginRenderPass(renderPassDescriptor);

  renderPass.setPipeline(graphicsPipeline);
  renderPass.setBindGroup(0, graphicsBindGroup);

  // draw 6 vertices
  // we are not providing any vertex data,
  // which are hard coded in the vertex shader.
  renderPass.draw(6, 1, 0, 0);

  renderPass.end();

  device.queue.submit([renderCommandEncoder.finish()]);
```
The graphics pipeline setup is more involved than the compute pipeline, because it has more moving parts, but the idea is the same. We create a pipeline, bind the resources to the pipeline, and submit the command to the queue.

If all goes well, you will get a screen with a faded gradient color on your screen.

## Draw a triangle

Now that we have the basic routine setup, let's feed a triangle to the compute shader and rasterize it on the screen.

First, we need to create a vertex buffer. The vertex buffer contains the vertex data we want to draw. In this case, we want to draw a triangle, so we need to create a vertex buffer with 3 vertices.

```ts
  const vertices = new Float32Array([
      10, 10,
      10, 80,
      80, 10
  ]);

  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  });
  new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
  vertexBuffer.unmap();
```
we also need to pass screen width and height to the compute shader, so we need to update the uniform buffer, and the bind group layout and bind group object.

```ts
  const uniformBufferSize = 4 * 2;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([width, height])
  );


 const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },

      /// NEW!!!
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage"
        }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform"
        }
      }
    ]
  });

  // create bind group
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: colorBuffer
        }
      },
      {
        binding: 1,
        resource: {
          buffer: vertexBuffer
        }
      },
      {
        binding: 2,
        resource: {
          buffer: uniformBuffer
        }
      }
    ]
  });

```

Now we need to update the compute shader to read the vertex data from the vertex buffer.

```wgsl
// create a draw line and draw pixel function

fn draw_pixel(x: u32, y: u32, r: u32, g: u32, b: u32) {
  let index = u32(x + y * u32(uniforms.screenWidth)) * 3;
  color_buffer.values[index] = r;
  color_buffer.values[index + 1] = g;
  color_buffer.values[index + 2] = b;
}


// bresenham's line algorithm
fn draw_line(v1: vec2<f32>, v2: vec2<f32>) {
  let dx = v2.x - v1.x;
  let dy = v2.y - v1.y;

  let steps = max(abs(dx), abs(dy));

  let x_increment = dx / steps;
  let y_increment = dy / steps;

  for(var i = 0u; i < u32(steps); i = i + 1) {
    let x = u32(v1.x + f32(i) * x_increment);
    let y = u32(v1.y + f32(i) * y_increment);
    draw_pixel(x, y, 255, 255, 255);
  }
}

```

you can find more information about bresenham's line algorithm [here](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).


You should see a triangle in wireframe mode on the screen. My implementation can be found [here](https://github.com/eddychu/webgpu-rasterizer.git)

We can extend this basic example to fill the triangle with color and add vertex transformation and camera feature, even with pbr material and  lighting calculation. But that is for another day.


**Reference**

[How to Build a Compute Rasterizer with WebGPU](
https://github.com/OmarShehata/webgpu-compute-rasterizer/blob/main/how-to-build-a-compute-rasterizer.md
)

[WebGPU — All of the cores, none of the canvas](https://surma.dev/things/webgpu/)

[WebGPU Samples](https://austin-eng.com/webgpu-samples/)