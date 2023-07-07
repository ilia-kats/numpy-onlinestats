#ifndef RUNNINGSTATS_H
#define RUNNINGSTATS_H

#include <cstdint>

class RunningStats
{
public:
    RunningStats();
    void clear();
    void push(double x);
    double mean() const;
    double var() const;
    double stddev() const;
    double skewness() const;
    double kurtosis() const;

    friend RunningStats operator+(const RunningStats a, const RunningStats b);
    RunningStats &operator+=(const RunningStats &rhs);

private:
    uint64_t n;
    double M1, M2, M3, M4;
};

#endif
