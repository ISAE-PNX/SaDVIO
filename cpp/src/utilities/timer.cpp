#include "utilities/timer.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stack>

namespace isae { namespace timer {

static std::stack<std::chrono::high_resolution_clock::time_point> timers;

void tic(){
    timers.push(std::chrono::high_resolution_clock::now());
}

double toc(std::string s){
    double dt = silentToc();
    if(dt>=0)
        std::cout << s << " elapsed time : " << std::round(dt) << "ms" << std::endl;
    return dt;
}

double silentToc()
{
    if(timers.size()>0){
        auto t = timers.top();
        timers.pop();
        auto end = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end-t).count();
        return dt/1000000.;
    }
    std::cerr << "toc called without tic" << std::endl;
    return -1;
}

}

}
