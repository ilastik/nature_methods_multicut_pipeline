#pragma once
#ifndef MCM_SVG_HXX
#define MCM_SVG_HXX

#include <iostream>
#include <stdexcept>
#include <array>
#include <cassert>

template<class T, class S = std::size_t>
class SavingSVG {
public:
	typedef T value_type;
	typedef S size_type;
	typedef std::array<unsigned char,3> RGBValue;
	
	struct GrayColoriserFromProbability {
        void operator()(RGBValue& RGB, const value_type prob) const;
	};

	struct LinearProbabilityColoriser {
		const RGBValue rgbHigh = {{255,0,0}};
		const RGBValue rgbMid = {{255,255,0}};
		const RGBValue rgbLow = {{0,255,0}};
		
		void operator()(RGBValue& RGB, const value_type prob) const ;
	};

	struct EdgePathDrawer{
		void up(const value_type x, const value_type y,const RGBValue RGB,std::ostream& stream) const;
		void right(const value_type x, const value_type y,const RGBValue RGB,std::ostream& stream) const;
	};
	
	struct ContourPathDrawer{
		void up(const value_type x, const value_type y,const RGBValue RGB,std::ostream& stream) const;
		void right(const value_type x, const value_type y,const RGBValue RGB,std::ostream& stream) const;
	};

	SavingSVG(value_type minValue = 0, value_type maxValue = 1);

	/// \param edgeGetter must provide
	/// 	value_type up(size_type x, size_type y)
	/// 	value_type right(size_type x, size_type y)
	/// \param ioParameters must specify width_, height_, inputFileName_
	template<class EDGE_GETTER,class PATH_DRAWER = ContourPathDrawer, class COLORIZER = LinearProbabilityColoriser>
	void operator()(
		const EDGE_GETTER &,
		const size_type,
		const size_type,
		std::ostream&,
		const PATH_DRAWER& = ContourPathDrawer(),
		const COLORIZER& = LinearProbabilityColoriser(),
		std::string const& original_image = std::string()
	);
	
private:
    value_type minValue_;
    value_type maxValue_;
};


template<class T, class S>
inline void
SavingSVG<T, S>::GrayColoriserFromProbability::operator()(
	RGBValue& RGB,
	const value_type probability
) const {
	RGB[0] = RGB[1] = RGB[2] = probability * 255;
}

template<class T, class S>
inline void
SavingSVG<T, S>::LinearProbabilityColoriser::operator()(
	RGBValue& RGB,
	const value_type probability
) const {
	value_type highMult = 0;
	value_type lowMult = 0;
	if(probability>0.5) {
		highMult = (probability-0.5)*2;
	}
	else {
		lowMult = 1-2*probability;
	}
	RGB[0] = rgbHigh[0]*highMult + rgbMid[0]*(1-highMult-lowMult) + rgbLow[0]*lowMult;
	RGB[1] = rgbHigh[1]*highMult + rgbMid[1]*(1-highMult-lowMult) + rgbLow[1]*lowMult;
	RGB[2] = rgbHigh[2]*highMult + rgbMid[2]*(1-highMult-lowMult) + rgbLow[2]*lowMult;
}


template<class T, class S>
inline void
SavingSVG<T, S>::EdgePathDrawer::up(
	const value_type x,
	const value_type y,
	const RGBValue RGB,
	std::ostream& stream
) const {
	stream << "<line ""x1=\"" <<
		x+0.5f << "\" y1=\""<<
		y+0.5f << "\" x2=\"" <<
		x+0.5f << "\" y2=\""<<
		y+1.5f << "\" style=\"stroke: rgb("<< +RGB[0]<<", "<< +RGB[1]<<", "<< +RGB[2]<<"); stroke-width: 0.6pt;\" />\n";
}
template<class T, class S>
inline void
SavingSVG<T, S>::EdgePathDrawer::right(
	const value_type x,
	const value_type y,
	const RGBValue RGB,
	std::ostream& stream
) const {
	stream << "<line x1=\"" <<
		x+0.5f << "\" y1=\""<<
		y+0.5f << "\" x2=\"" <<
		x+1.5f << "\" y2=\""<<
		y+0.5f << "\" style=\"stroke: rgb("<< +RGB[0]<<", "<< +RGB[1]<<", "<< +RGB[2]<<"); stroke-width: 0.6pt;\" />\n";
}


template<class T, class S>
inline void
SavingSVG<T, S>::ContourPathDrawer::up(
	const value_type x,
	const value_type y,
	const RGBValue RGB,
	std::ostream& stream
) const {
	stream << "<line x1=\"" <<
		x+0.0f << "\" y1=\""<<
		y+1.0f << "\" x2=\"" <<
		x+1.0f << "\" y2=\""<<
		y+1.0f << "\" style=\"stroke: rgb("<< +RGB[0]<<", "<< +RGB[1]<<", "<< +RGB[2]<<"); stroke-width: 0.6pt;\" />\n";
}

template<class T, class S>
inline void
SavingSVG<T, S>::ContourPathDrawer::right(const value_type x, const value_type y,const RGBValue RGB,std::ostream& stream) const {
	stream << "<line x1=\"" <<
		x+1.0f << "\" y1=\""<<
		y+0.0f << "\" x2=\"" <<
		x+1.0f << "\" y2=\""<<
		y+1.0f << "\" style=\"stroke: rgb("<< +RGB[0]<<", "<< +RGB[1]<<", "<< +RGB[2]<<"); stroke-width: 0.6pt;\" />\n";
}

template<class T, class S>
SavingSVG<T, S>::SavingSVG(
    value_type minValue,
    value_type maxValue
)   :minValue_(minValue), maxValue_(maxValue) {}

template<class T, class S>
template<class EDGE_GETTER, class PATH_DRAWER, class COLORIZER>
void
SavingSVG<T, S>::operator()(
	const EDGE_GETTER& edgeProbabilityGetter,
	const size_type width,
	const size_type height,
	std::ostream& stream,
	const PATH_DRAWER& pathDrawer,
	const COLORIZER& colorizer,
	std::string const& original_image
) {
	typedef T value_type;
	typedef S size_type;
	typedef std::array<unsigned char,3> RGBValue;
	
	if(width == 0 || height == 0 ){
		throw std::runtime_error("Invalid shape specified.");
	}
	
	stream << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
		"<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
	    "<svg version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" "
		"xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"" << 
		width << "\" height=\"" <<
		height << "\">\n";

	if (!original_image.empty())
		stream << "<image xlink:href=\"" << original_image << "\" x=\"0\" y=\"0\" height=\"" << height << "\" width=\"" << width << "\"/>\n";
	
	stream << "<!-- vertical edges -->\n";
	
	for(size_type x=0; x<width; ++x) {
        for(size_type y=0; y < height-1; ++y) {
			RGBValue RGB;
            value_type probability = edgeProbabilityGetter.up(x, y);
            assert(probability>=static_cast<value_type>(0) && probability<= static_cast<value_type>(1));
            if (probability>=minValue_ && probability <= maxValue_) {
                colorizer(RGB, probability);
				pathDrawer.up(x,y,RGB,stream);
            }
        }
    }
    stream << "<!-- horizontal edges -->" << std::endl;
    for(size_type x=0; x<width-1; ++x) {
        for(size_type y=0; y<height; ++y) {
			RGBValue RGB;
            value_type probability = edgeProbabilityGetter.right(x, y);
            assert(probability>=static_cast<value_type>(0) && probability<= static_cast<value_type>(1));
            if (probability>=minValue_ && probability <= maxValue_) {
                colorizer(RGB, probability);
				pathDrawer.right(x,y,RGB,stream);
            }
        }
    }
    stream << "</svg>" << std::endl;
}

#endif // #ifndef MCM_SVG_HXX