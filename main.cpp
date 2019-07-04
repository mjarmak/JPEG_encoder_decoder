#include <iostream>
#include <math.h>
#include <fstream>
#include <cmath>
#include <random>
#include <windows.h>
#include <errno.h>
#include <string.h>
#include <algorithm>
#include <iomanip>	// for the setw function
#include <memory>	// for auto_ptr

using namespace std;

int length_temp;

void saveImage(float* image,int bytes, string name){
    ofstream ofs(name,std::ios::binary);
    if(!ofs){
        cout << "Failed to save image"<< endl;
    }else{
        ofs.write((char*) image, bytes);
        ofs.close();
    }
}

void saveImage8bpp(unsigned int* image,int bytes, string name){
    ofstream ofs(name,std::ios::binary);
    if(!ofs){
        cout << "Failed to save image"<< endl;
    }else{
        ofs.write((char*) image, bytes);
        ofs.close();
    }
}

float *loadImage(string Imagename)
{
    ifstream image(Imagename.c_str(), ios::binary);
    float* ImageMat = new float[256*256];
    float value;
    int i=0;
    char buf[sizeof(float)];
    while (image.read(buf, sizeof(buf)))
    {
        memcpy(&value, buf, sizeof(value));

        ImageMat[i]  =  value;
        i++;
    }
    //cout << endl << "Total count "<< i<< endl ;
    return ImageMat;
}

float *loadText(string Imagename)
{
    ifstream image(Imagename.c_str(), ios::binary);
    float* ImageMat = new float[256*256];
    float value;
    int i=0;
    char buf[sizeof(float)];
    while (image.read(buf, sizeof(buf)))
    {
        memcpy(&value, buf, sizeof(value));

        ImageMat[i]  =  value;
        i++;
    }
    //cout << endl << "Total count "<< i<< endl ;
    return ImageMat;
}

float* generateCosineImage(int r){
    float* image = new float[r*r];
    float pi = 3.14159265358979323846;

    for (int x = 0; x < r; x++)  {
     for (int y = 0; y < r; y++) {
            image[256*x+y] = 0.5 + 0.5*cos(x*pi/32)*cos(y*pi/64);
     }
    }
    return image;
}
float* generateRandomUniformImage(int xSize, int ySize, int min, int max){
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(min,max);
    float* image = new float[xSize*ySize];

    for (int x = 0; x < xSize; x++)  {
     for (int y = 0; y < ySize; y++) {
            image[256*x+y] = (float)distribution(generator)/256;
     }
    }
    return image;
}
float* generateRandomNormalImage(int xSize, int ySize, float mean, float std){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean,std);
    float* image = new float[xSize*ySize];

    for (int x = 0; x < xSize; x++)  {
     for (int y = 0; y < ySize; y++) {
            image[256*x+y] = (float)distribution(generator);
     }
    }
    return image;
}
float *apply_filter(float image[256*256]){ //Gaussian filter
    int height = 256;
    int width = 256;
    int filterHeight = 3;
    int filterWidth = 3;
    int newImageHeight = height+filterHeight-1;
    int newImageWidth = width+filterWidth-1;
    int i,j,h,w;
    float temp = 0;
    float* blurred_image_data = new float[256*256];
    float* blurred_image_data_padded = new float[258*258];
    float* temp_widened_image = new float[258*258];

    float filter[3*3] = {0.0947416, 0.118318, 0.0947416, 0.118318 ,0.147761, 0.118318, 0.0947416, 0.118318, 0.0947416};
    for (i=0 ; i<258 ; i++) { // pad the image
        for (j=0 ; j<258 ; j++) {
            if ( i > 0 && i <257 && j>0 && j<257)
                temp_widened_image[i + (258*j)] = image[(i-1) + (256*(j-1))];
            else{
                if(i < 256 && j == 0)
                    temp_widened_image[i] = image[i];
                else if(i < 256 && j == 257)
                    temp_widened_image[i+257*258] = image[i+256*255];
                else if(i == 0 && j < 256)
                    temp_widened_image[258*j] = image[256*j];
                else if(i == 257 && j < 256)
                    temp_widened_image[257+258*j] = image[255+256*j];

                else if(i == 0 && j == 0)
                    temp_widened_image[0] = image[0];
                else if(i == 0 && j == 257)
                    temp_widened_image[257*258] = image[255*256];
                else if(i == 257 && j == 0)
                    temp_widened_image[257] = image[255];
                else if(i == 257 && j == 257)
                    temp_widened_image[257+257*258] = image[255+255*256];

            }
        }
    }
    //Initialize the image to zero
    for (i=0 ; i<258 ; i++)
        for (j=0 ; j<258 ; j++)
            blurred_image_data[i+ j*258] =0;
    //Apply filter
    for (i=1 ; i<257; i++) {
        for (j=1 ; j<257; j++) {
            float temp1 = 0;
            int h =0;
            int w =0;
            for (h=0; h<3 ; h++) {
                for (w=0 ; w<3 ; w++) {

                    if (i >0 && i<257 && j>0 && j<257){
                        temp1 += filter[w + (3*h)]*temp_widened_image[(i+h-1) + 258*(j+w-1)];
                    }
                    else
                        temp1 = 0;
               }
            }
            blurred_image_data_padded[(i) + 258*(j)] = temp1;
        }
    }
    for (i=0 ; i<256; i++) {
        for (j=0 ; j<256; j++) {
                blurred_image_data[i + 256*j] = blurred_image_data_padded[(i+1) + 258*j];
        }
    }
    return blurred_image_data;
}


float *add_images(float ImageMat1[256*256], float ImageMat2[256*256], int r, float weight)
{
    float* final_image_data = new float[r*r];
    for( int i = 0; i < r; i = i + 1 ){
            for( int j = 0; j < r; j = j + 1){
                final_image_data[i + r *j]  =  ImageMat1[i + r *j] + weight*ImageMat2[i + r *j];
            }
        }
    return final_image_data;
}

float *add_offset(float ImageMat1[256*256], float offset)
{
    float* final_image_data = new float[256*256];
    for( int i = 0; i < 256; i = i + 1 ){
            for( int j = 0; j < 256; j = j + 1){
                final_image_data[i + 256 *j]  =  ImageMat1[i + 256 *j] + offset;
            }
        }
    return final_image_data;
}

float PSNR(float* ImageMat_free,float* ImageMat_noisy, int max_value, int r){
    float mse = 0;
    float mse_temp =0;
    float mse_temp_1 =0;
    float psnr_value;
    for( int i = 0; i < r; i++ ){
            for( int j = 0; j < r; j++ ){
                mse_temp_1  =  ImageMat_free[i + r *j] - ImageMat_noisy[i + r *j];
                mse_temp = mse_temp_1*mse_temp_1;
                mse = mse+ mse_temp;
            }
        }
    mse = mse/(r*r);

    psnr_value = (20 * log10(max_value)) - (10* log10(mse));
    return psnr_value;
}

float *generate_DCT(int r){
    float* DCT = new float[r*r];
    float pi = 3.14159265358979323846;
    float c;
    for( int i = 0; i < r; i++){
            for( int j = 0; j < r; j++){
                if(j==0)
                    c = 1.0/sqrt(r);
                else
                    c = sqrt(2.0/r);

            DCT[i + r *j]  =  c * cos((pi/r)*(i + 0.5)*j);
            }
        }
    return DCT;
}


float *apply_DCT_row(float* ImageMat, float* DCT_mat, int r){
    float* dct = new float[r*r];
    float pi =3.14159265358979323846;
    float sum;

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
            sum = 0;
            for (int k = 0; k < r; k++){
                sum += ImageMat[r*i + k] * DCT_mat[r*j + k];
            }
            dct[r*i + j] = sum;
        }
    }
    return dct;
}

float *apply_DCT_col(float* ImageMat, float* DCT_mat, int r){
    float* dct = new float[r*r];
    float pi =3.14159265358979323846;
    float sum;

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
            sum = 0;
            for (int k = 0; k < r; k++){
                sum += ImageMat[i + r*k] * DCT_mat[r*j + k];
            }
            dct[i + r*j] = sum;
        }
    }
    return dct;
}


float *apply_IDCT_row(float* post_DCT, float* DCT_trans_mat, int r){
    float* inv_dct = new float[r*r];
    float pi =3.14159265358979323846;
    float sum;

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
            sum = 0;
            for (int k = 0; k < r; k++){
                sum += post_DCT[r*i + k] * DCT_trans_mat[r*j + k];
            }
            inv_dct[r*i + j] = sum;
        }
    }
    return inv_dct;
}
float *apply_IDCT_col(float* post_DCT, float* DCT_trans_mat, int r){
    float* inv_dct = new float[r*r];
    float pi =3.14159265358979323846;
    float sum;

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
            sum = 0;
            for (int k = 0; k < r; k++){
                sum += post_DCT[i + r*k] * DCT_trans_mat[r*j + k];
            }
            inv_dct[i + r*j] = sum;
        }
    }
    return inv_dct;
}

float *apply_DCT_8x8(float* image,float* basis_dct,int r){

    float* block = new float[8*8];
    float* temp = new float[8*8];
    float* dct = new float[r*r];

    int index_start;
    for(int i = 0;i < (r/8);i++){
        for(int j = 0;j < (r/8);j++){

             index_start = i*(8*r)+j*8;

             for(int x = 0;x <8;x++){
                for(int y = 0;y<8;y++){
                    block[x*8+y] = image[index_start+x*r+y];
                }
             }
             temp = apply_DCT_row(block,basis_dct,8);
             temp = apply_DCT_col(temp,basis_dct,8);
             for(int x = 0;x<8;x++){
                 for(int y=0;y<8;y++){
                     dct[index_start+x*r+y] = temp[x*8+y];//for dct image
                 }
             }
        }
    }
    return dct;
}

float *apply_IDCT_8x8(float* image,float* basis_idct, int r){

    float* block = new float[8*8];
    float* temp = new float[8*8];
    float* idct = new float[r*r];

    int index_start;
    for(int i = 0;i < (r/8);i++){
        for(int j = 0;j < (r/8);j++){

             index_start = i*(8*r)+j*8;

             for(int x = 0;x <8;x++){
                for(int y = 0;y<8;y++){
                    block[x*8+y] = image[index_start+x*r+y];
                }
             }
             temp = apply_IDCT_row(block,basis_idct,8);
             temp = apply_IDCT_col(temp,basis_idct,8);
             for(int x = 0;x<8;x++){
                 for(int y=0;y<8;y++){
                    idct[index_start+x*r+y] = round(temp[x*8+y]);//for idct image
                 }
             }
        }
    }
    return idct;
}

float *apply_Q_8x8(float* image,float* basis_q, int r){

    long* block = new long[8*8];
    float* temp = new float[8*8];
    float* q = new float[r*r];

    int index_start;
    for(int i = 0;i < (r/8);i++){
        for(int j = 0;j < (r/8);j++){

            index_start = i*(8*r)+j*8;

             for(int x = 0;x <8;x++){
                for(int y = 0;y<8;y++){
                    block[x*8+y] = image[index_start+x*r+y];
                }
             }
             for(int x = 0;x<8;x++){
                for(int y=0;y<8;y++){
                    block[x*8+y] = round(block[x*8+y]/basis_q[x*8+y]);
                    q[index_start+x*r+y] = block[x*8+y];
                }
             }
        }
    }
    return q;
}

float *apply_IQ_8x8(float* image,float* basis_q, int r){

    float* block = new float[8*8];
    float* temp = new float[8*8];
    float* iq = new float[r*r];

    int index_start;
    for(int i = 0;i < (r/8);i++){
        for(int j = 0;j < (r/8);j++){
            index_start = i*(8*r)+j*8;

             for(int x = 0;x <8;x++){
                for(int y = 0;y<8;y++){
                    block[x*8+y] = image[index_start+x*r+y];
                }
             }
             for(int x = 0;x<8;x++){
                for(int y=0;y<8;y++){
                    block[x*8+y] = block[x*8+y]*basis_q[x*8+y];
                    iq[index_start+x*r+y] = block[x*8+y];
                }
             }
        }
    }
    return iq;
}

float *transpose(float* Mat, int r){
    float* trans = new float[r*r];
    int i, j;

    // Finding transpose of matrix a[][] and storing it in array trans[][].
    for(int i = 0; i < r; ++i){
        for(int j = 0; j <r; ++j){
            trans[j + r*i]=Mat[i + r*j];
        }
    }
    return trans;
 }


float *threshold(float* Mat, float T, int r){
    float* Mat_thresh = new float[r*r];
    int i, j;

    // Finding transpose of matrix a[][] and storing it in array trans[][].
    for(int i = 0; i < r; ++i){
        for(int j = 0; j < r; ++j){
            if (Mat[i + r*j] < T && Mat[i + r*j] > -T)
                Mat_thresh[i+r*j]=0.0;
            else
                Mat_thresh[i+r*j] = Mat[i+r*j];

            }
        }
    return Mat_thresh;
 }

float *threshold_min_max(float* image, float min, float max, int r){
    float* t = new float[r*r];
    int i, j;

    // Finding transpose of matrix a[][] and storing it in array trans[][].
    for(int i = 0; i < r; ++i){
        for(int j = 0; j < r; ++j){
            if (image[i + r*j] < min)
                t[i+r*j]=0.0;
            else if(image[i + r*j] > max)
                t[i+r*j] = 255;
            else
                t[i+r*j] = image[i+r*j];
            }
        }
    return t;
 }
void *grayscale8bpp(float* image, float min, float max, int r, string name, int bytes){
    unsigned char* g = new unsigned char[r*r];
    int i, j;

    // Finding transpose of matrix a[][] and storing it in array trans[][].
    for(int i = 0; i < r; ++i){
        for(int j = 0; j < r; ++j){
            if (image[i + r*j] < min)
                g[i+r*j]=0.0;
            else if(image[i + r*j] > max)
                g[i+r*j] = 255;
            else
                g[i+r*j] = image[i+r*j];
            }
        }

    ofstream ofs(name,std::ios::binary);
    if(!ofs){
        cout << "Failed to save image"<< endl;
    }else{
        ofs.write((char*) g, bytes);
        ofs.close();
    }
 }

float *multiply_images(float image_1[256*256], float image_2[256*256])
{
    float* mult = new float[256*256];
    for(int i = 0; i < 256; i++)
        for(int j = 0; j < 256; j++)
            for(int k = 0; k < 256; k++)
            {
                mult[i+256*j] += image_1[i+256*k] * image_2[k+256*j];
            }
    return mult;
}

float* delta_encode(float* Mat, int r){
    float* image_out = new float[(r/8)*(r/8)];
    float last = 0;

    for(int i = 0; i < (r/8); i++){
        for(int j = 0; j < (r/8); j++){
            float current = Mat[r*i*8+j*8];
            image_out[(r/8)*i+j] = current - last;
            last = current;
        }
    }
    return image_out;
 }

float* delta_decode_long(float* image, int r){
    float* image_out = new float[r*r];
    float last = 0;
    for(int i = 0; i < (r/8); i++){
        for(int j = 0; j < (r/8); j++){
            float current = image[(r/8)*i+j];
            image_out[(r*i*8)+(j*8)] = current + last;
            last = current + last;
        }
    }
    return image_out;
}
float* delta_decode_small(float* image, int r){
    float* image_out = new float[r*r];
    float last = 0;
    for(int i = 0; i < (r/8); i++){
        for(int j = 0; j < (r/8); j++){
            float current = image[(r/8)*i+j];
            image_out[(r/8)*i+j] = current + last;
            last = current + last;
        }
    }
    return image_out;
}

float* zigzag(float* image, int r){
    float* image_out = new float[r*r];int lastValue = r * r - 1;
	int currNum = 0;int currDiag = 0;int loopFrom;int loopTo;int i;int row;int col;
	do{
		if(currDiag < r){ // if doing the upper-left triangular half
			loopFrom = 0;
			loopTo = currDiag;
		}else{ // doing the bottom-right triangular half
			loopFrom = currDiag - r + 1;
			loopTo = r-1;
		}for(i = loopFrom; i <= loopTo; i++){
			if (currDiag % 2 == 0){ // want to fill upwards
				row = loopTo - i + loopFrom;
				col = i;
			}else{ // want to fill downwards
				row = i;
				col = loopTo - i + loopFrom;
			}
			image_out[r*row + col] = image[currNum];
			currNum++;
		}
		currDiag++;
	}
	while(currDiag<=lastValue);
	return image_out;
}
float* unzigzag(float* image, int r){
    float* image_out = new float[r*r];
	int lastValue = r * r - 1;
	int currNum = 0;int currDiag = 0;int loopFrom;int loopTo;int i;int row;int col;
	do{
		if(currDiag < r){
			loopFrom = 0;
			loopTo = currDiag;
		}else{
			loopFrom = currDiag - r + 1;
			loopTo = r-1;
		}for(i = loopFrom; i <= loopTo; i++){
			if (currDiag % 2 == 0){
				row = loopTo - i + loopFrom;
				col = i;
			}else{
				row = i;
				col = loopTo - i + loopFrom;
			}
			image_out[currNum] = image[r*row + col];
			currNum++;
		}
		currDiag++;
	}
	while(currDiag<=lastValue);
	return image_out;
}
float* rle_encode(float* image, int r){

    float* image_out = new float[2*r*r];
    int index = 0;
    int count_total = 0;

    image = zigzag(image,r);

    for(int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
            int count = 1;
            count_total++;
            image_out[index] = image[j+r*i];
            //cout<<image[j+r*i]<<" ";
            index++;
            while (image[j+r*i] == image[j+r*i+1] && i*r+j < r*r-1){
                count++;
                count_total++;
                j++;
                if(j >= r && i < r){
                    i++;
                    j = 0;
                    }
            }
            image_out[index] = count;
            index++;
        }
    }
    image_out[index] = 8888;
    index++;
    image_out[index] = 0;
    //length_temp = index;
    return image_out;
}
float* rle_decode(float* image, int r){

    float* image_out = new float[r*r*2];
    int index = 0;
    int count_total = 0;

    for(int i = 0; i < 2*r*r; i+=2){
        if(image[i] == 8888)
            break;
        /*else if(i % 8 == 0 && r >= 128){
            image_out[index] = image[i];
            index++;
        }*/
        else{
            int count = image[i+1];
            while (count > 0){
                image_out[index] = image[i];
                index++;
                count--;
                count_total++;
            }
        }
    }
    image_out = unzigzag(image_out,r);
    //length_temp = index;
    return image_out;
}
float* rle_encode_8x8(float* image, int r, int block_start){

    float* image_out_long = new float[2*r*r];
    int index = 0;

    float* block = new float[8*8];
    float* block_encoded = new float[2*8*8];
    float* temp = new float[8*8];

    int index_start;
    for(int i = 0;i < (r/8);i++){
        for(int j = 0;j < (r/8);j++){
            index_start = i*(8*r)+j*8;

            for(int x = 0;x <8;x++){
                for(int y = 0;y<8;y++){
                    block[x*8+y] = image[index_start+x*r+y];
                }
            }
            block_encoded = rle_encode(block,8);
            for(int x = block_start; x < 2*8*8; x++){// stuffing until the block eof is reached
                if(block_encoded[x] == 8888)
                    break;
                image_out_long[index] = block_encoded[x];
                index++;
            }
            image_out_long[index] = 9999;
            index++;
            image_out_long[index] = 0;
            index++;
        }
    }
    image_out_long[index] = 8888;
    index++;
    image_out_long[index] = 0;
    float* image_out = image_out_long;
    length_temp = index;
    return image_out;
}

float* rle_decode_8x8(float* image, int r, int block_start){

    float* image_out = new float[r*r];
    int index = block_start;

    float* block = new float[2*8*8];
    float* block_unzigzaged = new float[8*8];
    int index_start = 0;
    int index_out = 0;

    for(int i = 0; i < (r/8); i++){
        for(int j = 0; j < (r/8); j++){
             index_out = i*(8*r)+j*8;

             for(int x = 0; x < 2*8*8; x++){
                if(image[index_start+x] == 9999){
                    block[x] = image[index_start+x];
                    index_start=index_start+x+2;//skip 9999
                    break;
                }else{
                    block[x] = image[index_start+x];
                }
             }

            float* block_unzigzaged = rle_decode(block, 8);

            int temp = block_start;
            while(temp > 0){
                image_out[index_out] = 0;
                temp--;
            }

             for(int x = 0;x<8;x++){
                for(int y = block_start;y<8-block_start;y++){
                    image_out[index_out+x*r+y] = block_unzigzaged[x*8+y];
                }
             }
            index = 0;
        }
    }
    return image_out;
}

float* dr_decode(float* image, float* delta, int r){

    float* image_out = new float[2*r*r];
    int index = 0;
    int index_delta = 0;
    for(int i = 0; i < r*r; i++){

        if(i == 0 || image[i-2] == 9999){
            image_out[index] = delta[index_delta];
            index++;
            image_out[index] = 1;
            index++;

            index_delta++;
        }else if(image[i] == 8888)
            break;
        image_out[index] = image[i];
        index++;

    }
    return image_out;
}

void test_rle(){

    float ss[5*5] = {0,1,1,1,1, 2,2,3,3,3, 5,5,5,5,5, 7,7,8,9,9, 1,1,1,1,1};
    cout << "zigzag" <<endl;
    float* sssszz = zigzag(ss,5);
    for (int i = 0; i < 25; i++)
        cout << sssszz[i] << " ";

    cout <<endl;
    cout << "unzigzag" <<endl;
    float* ssssuz = unzigzag(sssszz,5);
    for (int i = 0; i < 25; i++)
        cout << ssssuz[i] << " ";
    cout <<endl;
    float* ss_rle_encode = rle_encode(ss, 5);
    cout << "encoded" <<endl;
    for (int i = 0; i < 256; i++)
        if(ss_rle_encode[i] == 9999)
            break;
        else
            cout << ss_rle_encode[i] << " ";
    cout <<endl;
    float* ss_rle_decode = rle_decode(ss_rle_encode, 5);
    cout << "decoded" <<endl;
    for (int i = 0; i < 5*5; i++)
        cout << ss_rle_decode[i] << " ";
}

 string golomb(signed number){

    if(number == 0){
        string code = "1";
        //cout<<code<<" ";
        return code;
    }else{
        int bin[32];
        int remainder;
        int i = 0;
        int s = 0;
        //cout<<number<<" ";
        number*=2;
        if(number<0){
            number*=-1;
            number++;
        }
        while (number!=0)
        {
            remainder = number%2;
            bin[i++] = remainder;
            number /= 2;
        }
        s = i;
        for(int j = s-1; j >= 0; j--){
            bin[s + j] = 0;
        }
        string code = "";
        for(int j = s*2-2; j >= 0 ; j--){
            if(bin[j] == 1)
                code += "1";
            else if(bin[j] == 0)
                code += "0";
        }
        //cout<<code<<" ";
        return code;
     }
}

 signed int inv_golomb(string bin){

    long long number_long = std::stoll(bin,nullptr,2);
    string temp_char = "";
    temp_char.push_back(bin.back());
    number_long/=2;
    if(temp_char.compare("1") == 0){
            //number_long--;
            number_long*=-1;
    }
    //cout<<number_long<<" ";
    signed int number = number_long;
    return number;
}

string tostr(long long input){
    string output = std::to_string(input);
    return output;
}
 signed int inv_golomb_long(long long number_long){

    string bin = tostr(number_long);
    //cout<<bin;
    string temp_char = "";
    temp_char.push_back(bin.back());
    number_long/=2;
    if(temp_char.compare("1") == 0){
            //number_long--;
            number_long*=-1;
    }
    //cout<<number_long<<" ";
    signed int number = number_long;
    return number;
}

long long power(int x, int power){
    long long result;
    result =1.0;
    for (int i = 1; i <= power ; i++)
    {
        result = result*x;
    }
    return(result);
}

float* loadText_normal(string Imagename, int length){
    ifstream myReadFile;
    myReadFile.open(Imagename);

    char singleCharacter;
    int index_out = 0;
    float* output = new float[length];
    int keep = 1;
    if (myReadFile.is_open()){
        while (keep == 1){
            string number = "";
            myReadFile.get(singleCharacter);

            while(singleCharacter != ' '){
                number += singleCharacter;
                myReadFile.get(singleCharacter);
            }
            output[index_out] = std::stof(number);
            index_out++;

            if(number.compare("8888") == 0 || index_out >= length)
                keep = 0;
        }
    }
    myReadFile.close();
    length_temp = index_out;
    return output;
}

float* loadText(string Imagename, int length){
    ifstream myReadFile;
    myReadFile.open(Imagename);

    char singleCharacter;
    int index = 0;
    int index_out = 0;
    float* output = new float[length];
    float* block = new float[64];
    int zeros = 0;
    int keep = 1;
    long long temp_pow = 0;
    string temp = "";
    if (myReadFile.is_open()){
        while (keep == 1){
            myReadFile.get(singleCharacter);

            if(singleCharacter == '1')
                block[index] = 1;
            else
                block[index] = 0;

            index++;
            if(singleCharacter == '0'){
                zeros++;
            }else{
                while(zeros >= 0){
                    temp += singleCharacter;
                    index++;
                    zeros--;
                    if(zeros >= 0)
                        myReadFile.get(singleCharacter);
                }
                output[index_out] = inv_golomb(temp);
                if(output[index_out] == 8888 || index_out >= length){
                    keep = 0;
                }
                index = 0;
                zeros = 0;
                temp = "";
                index_out++;

            }
        }
    }
    myReadFile.close();
    length_temp = index_out;
    return output;
}
void approximate(float* lena, int x, int y){
    float *image_dct_8x8 = generate_DCT(8);
    saveImage(image_dct_8x8, 8*8*32/8, "image_dct_blocks.raw");

    float *image_dct_8x8_transpose = transpose(image_dct_8x8, 8);
    saveImage(image_dct_8x8_transpose, x*y*32/8, "image_dct_blocks_transpose.raw");

    float *lena_dct_8x8 = apply_DCT_8x8(lena, image_dct_8x8, 256);
    saveImage(lena_dct_8x8, x*y*32/8, "lena_dct_blocks.raw");


    float image_quantization_table[8*8] = {16,11,10,16,24,40,51,61,
                                        12,12,14,19,26,58,60,55,
                                        14,13,16,24,40,57,69,56,
                                        14,17,22,29,51,87,80,62,
                                        18,22,37,56,68,109,103,77,
                                        24,35,55,64,81,104,113,92,
                                        49,64,78,87,103,121,120,101,
                                        72,92,95,98,112,100,103,99};

    float *lena_Q_8x8 = apply_Q_8x8(lena_dct_8x8, image_quantization_table, 256);
    saveImage(lena_Q_8x8, x*y*32/8, "lena_Q_blocks.raw");


    float* lena_IQ_8x8 = apply_IQ_8x8(lena_Q_8x8, image_quantization_table, 256);
    saveImage(lena_IQ_8x8, x*y*32/8, "lena_iq_blocks.raw");
    cout<<"IQ decoded"<<endl;

    float *lena_idct_8x8 = apply_IDCT_8x8(lena_IQ_8x8, image_dct_8x8_transpose,256);
    saveImage(lena_idct_8x8, x*y*32/8, "lena_idct_blocks.raw");
    cout<<"IDCT decoded"<<endl;
    }

void encode(float* lena, int x ,int y){
    float *image_dct_8x8 = generate_DCT(8);
    saveImage(image_dct_8x8, 8*8*32/8, "image_dct_blocks.raw");

    float *lena_dct_8x8 = apply_DCT_8x8(lena, image_dct_8x8, 256);
    saveImage(lena_dct_8x8, x*y*32/8, "lena_dct_blocks.raw");


    float image_quantization_table[8*8] = {16,11,10,16,24,40,51,61,
                                        12,12,14,19,26,58,60,55,
                                        14,13,16,24,40,57,69,56,
                                        14,17,22,29,51,87,80,62,
                                        18,22,37,56,68,109,103,77,
                                        24,35,55,64,81,104,113,92,
                                        49,64,78,87,103,121,120,101,
                                        72,92,95,98,112,100,103,99};

    float *lena_Q_8x8 = apply_Q_8x8(lena_dct_8x8, image_quantization_table, 256);
    saveImage(lena_Q_8x8, x*y*32/8, "lena_Q_blocks.raw");

    float* lena_delta_encode = delta_encode(lena_Q_8x8, 256);
    saveImage(lena_delta_encode, 32*32*32/8, "lena_delta_encode.raw");


    float* lena_dc = delta_decode_small(lena_delta_encode, 256);
    saveImage(lena_dc,32*32*32/8, "lena_dc.raw");

    std::ofstream outfile_lena_dc ("lena_dc.txt");
    for(int i = 0; i < 32*32; i++)
        outfile_lena_dc << lena_dc[i] << " " ;
    outfile_lena_dc.close();
    cout<<"DC components to .txt"<<endl;



    float* lena_rle_encode_8x8 = rle_encode_8x8(lena_Q_8x8, 256, 2);
    saveImage(lena_rle_encode_8x8, length_temp*32/8, "lena_rle_encode_blocks.raw");
    cout<<"RLE encoded"<<endl;

    std::ofstream outfile_lena_delta_encode_golomb ("lena_delta_encode_golomb.txt");
    for(int i = 0; i < 32*32; i++)
        outfile_lena_delta_encode_golomb << golomb(lena_delta_encode[i]) << "" ;
    outfile_lena_delta_encode_golomb.close();
    cout<<"Delta encoded to golomb to .txt"<<endl;

    std::ofstream outfile_lena_rle_encode_block_golomb ("lena_rle_encode_blocks_golomb.txt");
    for(int i = 0; i < length_temp; i++)
        outfile_lena_rle_encode_block_golomb << golomb(lena_rle_encode_8x8[i]) << "" ;
    outfile_lena_rle_encode_block_golomb.close();
    cout<<"RLE encoded to golomb to .txt"<<endl;

}

void decode(string name1, string name2, int x, int y){

    float *image_dct_8x8 = generate_DCT(8);
    saveImage(image_dct_8x8, 8*8*32/8, "image_dct_blocks.raw");

    float *image_dct_8x8_transpose = transpose(image_dct_8x8, 8);
    saveImage(image_dct_8x8_transpose, x*y*32/8, "image_dct_blocks_transpose.raw");

    float* lena_rle_encode_blocks_load_golomb = loadText(name1, 256*256*2);
    saveImage(lena_rle_encode_blocks_load_golomb,length_temp*32/8, "lena_rle_encode_blocks_load_golomb.raw");
    cout<<"RLE Read"<<endl;

    float* lena_delta_encode_load_golomb = loadText(name2, 32*32);
    cout<<"Delta Read"<<endl;
    float* lena_delta_decode_load_golomb = delta_decode_small(lena_delta_encode_load_golomb, 256);
    saveImage(lena_delta_decode_load_golomb,32*32*32/8, "lena_delta_decode_load_golomb.raw");
    cout<<"Delta Decoded"<<endl;

    float* lena_dr_decode_q_blocks_load_golomb = dr_decode(lena_rle_encode_blocks_load_golomb, lena_delta_decode_load_golomb, 256);
    cout<<"RLE & Delta combined"<<endl;
    float* lena_rle_decode_q_blocks_load_golomb = rle_decode_8x8(lena_dr_decode_q_blocks_load_golomb, 256, 0);
    cout<<"RLE Decoded"<<endl;

    float image_quantization_table[8*8] = {16,11,10,16,24,40,51,61,
                                        12,12,14,19,26,58,60,55,
                                        14,13,16,24,40,57,69,56,
                                        14,17,22,29,51,87,80,62,
                                        18,22,37,56,68,109,103,77,
                                        24,35,55,64,81,104,113,92,
                                        49,64,78,87,103,121,120,101,
                                        72,92,95,98,112,100,103,99};

    float* lena_rle_decode_iq_blocks_load_golomb = apply_IQ_8x8(lena_rle_decode_q_blocks_load_golomb, image_quantization_table, 256);
    float *lena_rle_decode_blocks_load_golomb = threshold_min_max(apply_IDCT_8x8(lena_rle_decode_iq_blocks_load_golomb, image_dct_8x8_transpose,256),0,255,256);
    saveImage(lena_rle_decode_blocks_load_golomb, x*y*32/8, "lena_rle_decode_blocks_idct_load_golomb.raw");

    grayscale8bpp(lena_rle_decode_blocks_load_golomb, 0, 255, 256, "lena_rle_decode_blocks_idct_load_golomb_grayscale8bpp.raw", x*y);
}

int main(){

    int x = 256;
    int y = 256;

    float image_quantization_table[8*8] = {16,11,10,16,24,40,51,61,
                                        12,12,14,19,26,58,60,55,
                                        14,13,16,24,40,57,69,56,
                                        14,17,22,29,51,87,80,62,
                                        18,22,37,56,68,109,103,77,
                                        24,35,55,64,81,104,113,92,
                                        49,64,78,87,103,121,120,101,
                                        72,92,95,98,112,100,103,99};
    saveImage(image_quantization_table, 8*8*32/8, "image_quantization_table.raw");


    float* image_cos = generateCosineImage(256);
    saveImage(image_cos,x*y*32/8, "image_cos.raw");

    //float* image_noise_uni = generateRandomUniformImage(x,y,0,255); //Mean = 0.501
    //saveImage(image_noise_uni,x*y*32/8, "image_noise_uni.raw");

    float* image_noise_03 = generateRandomNormalImage(x,y,0,10); //std = 0.3
    float* image_noise_06 = generateRandomNormalImage(x,y,0,20); //std = 0.6
    float* image_noise_d = generateRandomNormalImage(x,y,0,14.1); //std = d
    saveImage(image_noise_03,x*y*32/8, "image_noise_Gaussian_zeromean_10std.raw");
    saveImage(image_noise_06,x*y*32/8, "image_noise_Gaussian_zeromean_20std.raw");
    saveImage(image_noise_d,x*y*32/8, "image_noise_Gaussian_zeromean_14_1.std.raw");

    float *lena = loadImage("lena_256x256.raw");
    //float *image_noise = loadImage("noise_Gaussian_zeromean_0.289std.raw");

    float* lena_noisy_03 = add_images(lena, image_noise_03, 256, 1);
    saveImage(lena_noisy_03,x*y*32/8, "lena_noisy_10.raw");

    float* lena_noisy_06 = add_images(lena, image_noise_06, 256, 1);
    saveImage(lena_noisy_06,x*y*32/8, "lena_noisy_20.raw");

    float* lena_noisy_d = add_images(lena, image_noise_d, 256, 1);
    saveImage(lena_noisy_d,x*y*32/8, "lena_noisy_14_1.raw");

    float psnr;
    psnr = PSNR(lena, lena_noisy_03, 255, 256);
    cout<< "PSNR value of Normal vs Noisy 10: "<< psnr << endl;

    psnr = PSNR(lena, lena_noisy_06, 255, 256);
    cout<< "PSNR value of Normal vs Noisy 20: "<< psnr << endl;

    psnr = PSNR(lena, lena_noisy_d, 255, 256);
    cout<< "PSNR value of Normal vs Noisy 14.1: "<< psnr << endl;

    float *lena_noisy_d_blurred = apply_filter(lena_noisy_d);
    saveImage(lena_noisy_d_blurred,x*y*32/8, "lena_noisy_d_blurred.raw");

    psnr = PSNR(lena, lena_noisy_d_blurred, 255, 256);
    cout<< "PSNR value of Normal vs Noisy_d then Blurred: "<< psnr << endl;

    float *lena_blurred = apply_filter(lena);
    saveImage(lena_blurred,x*y*32/8, "lena_blurred.raw");
    float *lena_blurred_noisy_d = add_images(lena_blurred, image_noise_d, 256, 1);
    saveImage(lena_blurred_noisy_d,x*y*32/8, "lena_blurred_noisy_d.raw");

    psnr = PSNR(lena, lena_blurred_noisy_d, 255, 256);
    cout<< "PSNR value of Normal vs Blurred then Noisy_d: "<< psnr << endl;

    psnr = PSNR(lena, lena_blurred, 255, 256);
    cout<< "PSNR value of Normal vs Blurry: "<< psnr << endl;

    float *lena_sharp = add_images(lena, lena_blurred, 256, -1);
    saveImage(lena_sharp,x*y*32/8, "lena_sharp.raw");

    psnr = PSNR(lena, lena_sharp, 255, 256);
    cout<< "PSNR value of Normal vs Sharp: "<< psnr << endl;

    float *lena_sharpenned = threshold_min_max(add_images(lena, lena_sharp, 256, 10),0,255,256);
    saveImage(lena_sharpenned,x*y*32/8, "lena_sharpenned.raw");

    psnr = PSNR(lena, lena_sharpenned, 255, 256);
    cout<< "PSNR value of Normal vs Sharpenned: "<< psnr << endl;

    float *lena_blurred_sharp = threshold_min_max(add_images(lena_blurred, lena_sharp, 256, 0.5),0,255,256);
    saveImage(lena_blurred_sharp,x*y*32/8, "lena_blurred_sharp.raw");

    psnr = PSNR(lena, lena_blurred_sharp, 255, 256);
    cout<< "PSNR value of Normal vs Blurred then Sharp: "<< psnr << endl;

    approximate(lena, 256, 256);

    encode(lena,x,y);
    decode("lena_rle_encode_blocks_golomb.txt","lena_delta_encode_golomb.txt",x,y);

    return 0;
}
