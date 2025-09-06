#pragma once

#define lambda(return_type, function_body) ({ return_type __fn__ function_body __fn__; })
