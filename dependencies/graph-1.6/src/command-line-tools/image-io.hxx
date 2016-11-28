#pragma once
#ifndef TOOLS_IMAGE_IO_HXX
#define TOOLS_IMAGE_IO_HXX

#include <cassert>
#include <cstddef>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <iostream>

#include <andres/marray.hxx>

template<class T = unsigned char>
void
loadImagePPM(
    const std::string& fileName,
    andres::Marray<T>& image
) {
    std::FILE *aStream;
    aStream = fopen(fileName.c_str(),"rb");
    if (aStream == 0)
        std::cerr << "File not found: " << fileName << std::endl;
    int dummy;
    std::size_t width;
    std::size_t height;
    std::size_t channels;

// Find beginning of file (P6)
    while (getc(aStream) != 'P');
    dummy = getc(aStream);
    if (dummy == '5') channels = 1;
    else if (dummy == '6') channels = 3;
    do dummy = getc(aStream);
    while (dummy != '\n' && dummy != ' ');
    // Remove comments and empty lines
    dummy = getc(aStream);
    while (dummy == '#') {
        while (getc(aStream) != '\n');
        dummy = getc(aStream);
    }
    while (dummy == '\n')
        dummy = getc(aStream);
    // Read image size
    width = dummy-48;
    while ((dummy = getc(aStream)) >= 48 && dummy < 58)
        width = 10*width+dummy-48;
    while ((dummy = getc(aStream)) < 48 || dummy >= 58);
    height = dummy-48;
    while ((dummy = getc(aStream)) >= 48 && dummy < 58)
        height = 10*height+dummy-48;
    while (dummy != '\n' && dummy != ' ')
        dummy = getc(aStream);
    while (dummy < 48 || dummy >= 58) dummy = getc(aStream);
    while ((dummy = getc(aStream)) >= 48 && dummy < 58);
    if (dummy != '\n') while (getc(aStream) != '\n');
    // Adjust size of data structure
    std::size_t shape[] = {height,width, channels};
    image.resize(shape, shape + 3);
    // Read image data
    int aSize = width*height;
    if (channels == 1)
        for (int i = 0; i < aSize; i++)
            image(i) = getc(aStream);
    else {
        int aSizeTwice = aSize+aSize;
        for (int i = 0; i < channels*aSize; i+=3) {
            image(i) = getc(aStream);
            image(i+1) = getc(aStream);
            image(i+2) = getc(aStream);
        }
    }
    image.transpose(0,1);
    fclose(aStream);
}

inline void convertColor(size_t iColor, unsigned char *cColor)
{
    cColor[2] = (iColor >> 16) & 0xFF;
    cColor[1] = (iColor >> 8) & 0xFF;
    cColor[0] = iColor & 0xFF;
}

void writePPM(const char* filename, int W, int H, const unsigned char* data)
{
    FILE* fp = fopen ( filename, "w" );
    if ( !fp )
    {
        printf ( "Failed to open file '%s'!\n", filename );
    }
    fprintf ( fp, "P6\n%d %d\n%d\n", W, H, 255 );
    fwrite ( data, 1, W*H*3, fp );
    fclose ( fp );
}

#endif // #ifndef TOOLS_IMAGE_IO_HXX
