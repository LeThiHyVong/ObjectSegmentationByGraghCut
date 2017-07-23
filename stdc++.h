//
// Created by nvkhoi on 6/18/17.
//

#ifndef GRAPHCUTSEGMENTATION_STDC_H_H
#define GRAPHCUTSEGMENTATION_STDC_H_H

// standard template library

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <array>
#include <list>
#include <stack>
#include <queue>
#include <string>

#include <algorithm>
#include <utility>
#include <cstdint>
#include <cstdlib>
#include <functional>

#include <thread>
#include <mutex>
#include <memory>

// OpenCV framework
#include <opencv2/opencv.hpp>

// Boost framework


// User-defined
// dedicated/retrospective cast
template<typename T>
struct memfun_type
{
    using type = void;
};

template<typename Ret, typename Class, typename... Args>
struct memfun_type<Ret(Class::*)(Args...) const>
{
    using type = std::function<Ret(Args...)>;
};

template<typename F>
typename memfun_type<decltype(&F::operator())>::type
FFL(F const &func)
{ // Function from lambda !
    return func;
}

#endif //GRAPHCUTSEGMENTATION_STDC_H_H
