#coding=utf-8
import jittor as jt 

def nms_cpu(dets,scores,iou_threshold):
    assert dets.ndim==2 and dets.shape[1]==4 and scores.ndim==1 and scores.shape[0]==dets.shape[0]
    if dets.numel()==0:
        return jt.zeros((0,)).int32()

    x1,y1,x2,y2 = dets[:,0],dets[:,1],dets[:,2],dets[:,3]
    areas = (x2-x1)*(y2-y1)

    order,_ = scores.argsort(0,descending=True)

    dets_num = dets.shape[0]
    keep = jt.code((dets_num,),'bool',[x1,y1,x2,y2,areas,order],cpu_header=r'''
    #undef out
    #include <executor.h>
    #include <cmath>
    using namespace std;
    ''',cpu_src=f'double iou_threshold = {iou_threshold};'+r'''
  @alias(x1_,in0)
  @alias(y1_,in1)
  @alias(x2_,in2)
  @alias(y2_,in3)
  @alias(areas_,in4)
  @alias(order_,in5)
  @alias(keep_,out0)

  auto x1 = x1__p;
  auto y1 = y1__p;
  auto x2 = x2__p;
  auto y2 = y2__p;
  auto areas = areas__p;
  auto keep = keep__p;
  auto order = order__p;


  int ndets = x1__shape0;
  memset(keep, 0, ndets);

  int matrices_size = ndets*sizeof(uint8);
  size_t suppressed_allocation;
  auto suppressed = (uint8 *)exe.allocator->alloc(matrices_size, suppressed_allocation);
  memset(suppressed,0,matrices_size);

  for (int _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[i] = true;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max((x1__type)0, xx2 - xx1);
      auto h = std::max((x1__type)0, yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold)
        suppressed[j] = 1;
    }
   }
   exe.allocator->free(suppressed, matrices_size, suppressed_allocation);
    ''')
    return keep.where()[0]
    
  
def nms_cuda(dets,scores,iou_threshold):
    assert dets.ndim==2 and dets.shape[1]==4 and scores.ndim==1 and scores.shape[0]==dets.shape[0]
    if dets.numel()==0:
        return jt.zeros((0,)).int32()

    order,_ = scores.argsort(0,descending=True)
    dets_sorted = dets[order,:]
    dets_num  = dets.shape[0]

    keep = jt.code((dets_num,),'bool',[dets_sorted],cuda_header =r'''
#undef out
#include <vector>
#include <executor.h>
using namespace std;

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline bool devIoU(T const* const a, T const* const b, const float threshold) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  T interS = width * height;
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return (interS / (Sa + Sb - interS)) > threshold;
}

template <typename T>
__global__ void nms_kernel(
    int n_boxes,
    double iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = (n_boxes-1)/threadsPerBlock+1;
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
''',cuda_src=f'double iou_threshold={iou_threshold};'+r'''
  @alias(dets_sorted,in0)
  @alias(keep,out)

  int dets_num = dets_sorted_shape0;

  const int col_blocks = (dets_num-1)/threadsPerBlock+1;

  int matrices_size = dets_num * col_blocks*sizeof(unsigned long long);
  size_t mask_allocation;
  unsigned long long* mask_p = (unsigned long long*)exe.allocator->alloc(matrices_size, mask_allocation);

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  nms_kernel<dets_sorted_type><<<blocks, threads>>>(
            dets_num,
            iou_threshold,
            dets_sorted_p,
            mask_p);

   checkCudaErrors(cudaDeviceSynchronize());
   std::vector<unsigned long long> remv(col_blocks);
   memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
   memset(keep_p, 0, dets_num);

   auto keep_out = keep_p;

    for (int i = 0; i < dets_num; i++) {
       int nblock = i / threadsPerBlock;
       int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[i] = true;
      unsigned long long* p = mask_p + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
       }
      }
    }
   exe.allocator->free(mask_p, matrices_size, mask_allocation);
''')

    return order[keep]


def nms(dets,scores,iou_threshold):
  if jt.flags.use_cuda==1:
    return nms_cuda(dets,scores,iou_threshold)
  else:
    return nms_cpu(dets,scores,iou_threshold)
  
