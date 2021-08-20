// Adapted from Halide tutorial lesson 2.
#include <Halide.h>
using Halide::Image;
#include "../apps/support/image_io.h"

int main(int argc, char **argv)
{
    Halide::Image<uint8_t> input = load<uint8_t>(argv[1]);
    Halide::Func brighter;
    Halide::Var x, y, c;
    Halide::Expr value = input(x, y, c);

    value = Halide::cast<float>(value);
    value = value * 1.5f;
    value = Halide::min(value, 255.0f);
    value = Halide::cast<uint8_t>(value);

    brighter(x, y, c) = value;

    Halide::Image<uint8_t> output = brighter.realize(input.width(), input.height(), input.channels());

    save(output, "brighter.png");
    
    return 0;
}
