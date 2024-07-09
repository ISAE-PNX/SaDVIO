#ifndef TIMER_H
#define TIMER_H

#include <string>

namespace isae {

//< time measurement utils
namespace timer {

class timer_handle;

/**
 * @brief pushes the current time on the stack
 */
void tic();
/**
 * @brief pops a time from the stack and returns the difference with the current time
 * @return delta time (ms)
 */
double silentToc();

/**
 * @brief pops a time from the stack and prints the difference with the current time
 * @param s timer message
 * @return delta time (ms)
 */
double toc(std::string s="");

/**
 * @brief Handy way for timing an action
 * @author Adrien Debord
 *
 * Calls tic() on creation and toc() on destruction
 */
class timer_handle {
public:

    /**
     * @brief starts the timer
     * @param name timer message
     */
    timer_handle(std::string name){
        this->name=name;
        tic();
    }

    /**
     * @brief stops the timer
     * @return delta time (ms)
     */
    double toc(){
        if(!_toced){
            _toced = true;
            return timer::toc(name);
        }
        return -1;
    }

    ~timer_handle(){
        this->toc();
    }
private :
    bool _toced = false; //< has already been stoped ?
    std::string name; //< timer message
    timer_handle(timer_handle &)=delete;
};


}}
#endif // TIMER_H
