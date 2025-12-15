# CMake generated Testfile for 
# Source directory: /Users/sueszli/dev/autograd.c
# Build directory: /Users/sueszli/dev/autograd.c/debug_build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_activations]=] "/Users/sueszli/dev/autograd.c/debug_build/test_activations_binary")
set_tests_properties([=[test_activations]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_activations_backward]=] "/Users/sueszli/dev/autograd.c/debug_build/test_activations_backward_binary")
set_tests_properties([=[test_activations_backward]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_arithmetic]=] "/Users/sueszli/dev/autograd.c/debug_build/test_arithmetic_binary")
set_tests_properties([=[test_arithmetic]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_augment]=] "/Users/sueszli/dev/autograd.c/debug_build/test_augment_binary")
set_tests_properties([=[test_augment]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_convolutions]=] "/Users/sueszli/dev/autograd.c/debug_build/test_convolutions_binary")
set_tests_properties([=[test_convolutions]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_convolutions_backward]=] "/Users/sueszli/dev/autograd.c/debug_build/test_convolutions_backward_binary")
set_tests_properties([=[test_convolutions_backward]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_layers]=] "/Users/sueszli/dev/autograd.c/debug_build/test_layers_binary")
set_tests_properties([=[test_layers]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_losses]=] "/Users/sueszli/dev/autograd.c/debug_build/test_losses_binary")
set_tests_properties([=[test_losses]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_losses_backward]=] "/Users/sueszli/dev/autograd.c/debug_build/test_losses_backward_binary")
set_tests_properties([=[test_losses_backward]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
add_test([=[test_optimizers]=] "/Users/sueszli/dev/autograd.c/debug_build/test_optimizers_binary")
set_tests_properties([=[test_optimizers]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/sueszli/dev/autograd.c/CMakeLists.txt;160;add_test;/Users/sueszli/dev/autograd.c/CMakeLists.txt;0;")
subdirs("_deps/unity-build")
