#pragma once
#ifndef TOOLS_FUNCTIONAL_HXX
#define TOOLS_FUNCTIONAL_HXX

#include <cassert>

template<typename T, typename U>
struct NegativeLogProbabilityToInverseProbability {
    typedef T argument_type;
    typedef U result_type;

    result_type operator()(argument_type x) const {
        return 1-::exp(-x );
    }
};

template<typename T, typename U>
struct ProbabilityToNegativeLogInverseProbability {
    typedef T argument_type;
    typedef U result_type;

    result_type operator()(argument_type x) const {
        return -::log( 1-x );
    }
};

template<typename T, typename U>
struct ProbabilityToLogit {
    typedef T argument_type;
    typedef U result_type;

    result_type operator()(argument_type x) const {
      return ::log( (1-x)/x );
    }
};
#endif // #ifndef TOOLS_FUNCTIONAL_HXX
