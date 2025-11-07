#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdlib.h> // Added for malloc/free
#include <pthread.h> // Added for pthreads
#include <unistd.h>  // Added for sysconf (to get number of cores)
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};


//getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
//Paramters: srcImage:  An Image struct populated with the image being convoluted
//           x: The x coordinate of the pixel
//           y: The y coordinate of the pixel
//           bit: The color channel being manipulated
//           algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my,i,span;
    span=srcImage->width*srcImage->bpp;
    // for the edge pixes, just reuse the edge pixel
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;
    double result_double =
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];

    // Clamping the result to the 0-255 range for uint8_t
    if (result_double > 255) result_double = 255;
    if (result_double < 0) result_double = 0;
    
    return (uint8_t)result_double;
}


// A struct to pass data to each thread.
// Each thread needs to know which rows it's responsible for.
typedef struct {
    Image* srcImage;
    Image* destImage;
    Matrix algorithm;
    int start_row;
    int end_row;
} ThreadData;

// The worker function that each thread will execute.
// It computes the convolution for its assigned "slice" of the image.
void* convolute_thread_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int row, pix, bit;

    // Loop over the assigned rows, only
    for (row = data->start_row; row < data->end_row; row++) {
        for (pix = 0; pix < data->srcImage->width; pix++) {
            for (bit = 0; bit < data->srcImage->bpp; bit++) {
                // Calculate the new pixel value
                uint8_t value = getPixelValue(data->srcImage, pix, row, bit, data->algorithm);
                // Write the result to the destination image
                data->destImage->data[Index(pix, row, data->srcImage->width, bit, data->srcImage->bpp)] = value;
            }
        }
    }
    return NULL;
}

//convolute:  Applies a kernel matrix to an image (Parallel Version)
//This function now acts as the manager that creates and joins threads.
void convolute(Image* srcImage,Image* destImage,Matrix algorithm){
    
    // Get the number of available CPU cores
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int num_threads = (int)num_cores;
    printf("Using %d threads.\n", num_threads); 

    // Allocate memory for thread identifiers and thread data
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));

    if (threads == NULL || thread_data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for threads.\n");
        // Fallback to serial execution if allocation fails
        for (int row = 0; row < srcImage->height; row++) {
            for (int pix = 0; pix < srcImage->width; pix++) {
                for (int bit = 0; bit < srcImage->bpp; bit++) {
                    destImage->data[Index(pix, row, srcImage->width, bit, srcImage->bpp)] = getPixelValue(srcImage, pix, row, bit, algorithm);
                }
            }
        }
        free(threads);
        free(thread_data);
        return;
    }

    // Calculate how many rows each thread will process
    int rows_per_thread = srcImage->height / num_threads;
    int remaining_rows = srcImage->height % num_threads;
    int current_row = 0;

    // Launch threads
    for (int i = 0; i < num_threads; i++) {
        // Assign data to this thread
        thread_data[i].srcImage = srcImage;
        thread_data[i].destImage = destImage;
        memcpy(thread_data[i].algorithm, algorithm, sizeof(Matrix)); 
        
        thread_data[i].start_row = current_row;
        
        // Distribute the remaining rows one by one to the first few threads
        int rows_for_this_thread = rows_per_thread;
        if (remaining_rows > 0) {
            rows_for_this_thread++;
            remaining_rows--;
        }
        
        thread_data[i].end_row = current_row + rows_for_this_thread;
        current_row = thread_data[i].end_row;

        // Create the thread
        int rc = pthread_create(&threads[i], NULL, convolute_thread_worker, (void*)&thread_data[i]);
        if (rc) {
            fprintf(stderr, "Error: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Free the memory we allocated
    free(threads);
    free(thread_data);
}




//Usage: Prints usage information for the program
//Returns: -1
int Usage(){
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY, which does nothing but copy the image.
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

//main:
//argv is expected to take 2 arguments.  First is the source file name (can be jpg, png, bmp, tga).  Second is the lower case name of the algorithm.
int main(int argc,char** argv){
    long t1,t2;
    t1=time(NULL);
    printf("This is the start time: %ld\n", t1);

    stbi_set_flip_vertically_on_load(0);  
    if (argc!=3) return Usage();
    char* fileName=argv[1];
    if (!strcmp(argv[1],"pic4.jpg")&&!strcmp(argv[2],"gauss")){
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }
    enum KernelTypes type=GetKernelType(argv[2]);

    Image srcImage,destImage,bwImage;   
    srcImage.data=stbi_load(fileName,&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file %s.\n",fileName);
        return -1;
    }
    destImage.bpp=srcImage.bpp;
    destImage.height=srcImage.height;
    destImage.width=srcImage.width;
    destImage.data=malloc(sizeof(uint8_t)*destImage.width*destImage.bpp*destImage.height);
    
    // This call now points to our parallelized convolute function
    convolute(&srcImage,&destImage,algorithms[type]);
    
    stbi_write_png("output.png",destImage.width,destImage.height,destImage.bpp,destImage.data,destImage.bpp*destImage.width);
    stbi_image_free(srcImage.data);
    
    free(destImage.data);
    t2=time(NULL);
    printf("This is the end time: %ld\n", t2);
    printf("Took %ld seconds\n",t2-t1);
   return 0;
}