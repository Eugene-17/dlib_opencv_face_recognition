# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# [EDIT] Set Path to OpenCV and include the directories
# pathToOpenCV is just an example to how to write in Mac.
# General format: /Users/user-name/OpenCV-android-sdk/sdk/native
#Duong dan den OpenCV include
set(pathToOpenCV D:/GR2/OpenCV-android-sdk/sdk/native)
include_directories(${pathToOpenCV}/jni/include)

#add thu vien lib_opencv
add_library(lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION D:/GR2/dlib_opencv_face_recognition/app/src/main/jniLibs/opencvLibs/${ANDROID_ABI}/libopencv_java3.so)

# Duong dan den thu muc dlib include
#set include directories for dlib
include_directories(D:/GR2/dlib_opencv_face_recognition/app/src/main/jniLibs/dlib/include)

# EDIT add dlib
# Link to libdlib.so.
# Shared lib will also be tucked into APK and sent to target
# refer to app/build.gradle, jniLibs section for that purpose.
# ${ANDROID_ABI} is handy for our purpose here. Probably this ${ANDROID_ABI} is
# the most valuable thing of this sample, the rest are pretty much normal cmake
# add thu vien dlib
add_library(lib_dlib SHARED IMPORTED)
set_target_properties(lib_dlib PROPERTIES IMPORTED_LOCATION D:/GR2/dlib_opencv_face_recognition/app/src/main/jniLibs/dlib/lib/${ANDROID_ABI}/libdlib.so)


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
# Build project shared lib
# Build shared lib cho project tu file ma nguon cpp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -std=c++11")
add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        src/main/cpp/native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

#link cac lib duoc su dung vao native-lib
target_link_libraries( # Specifies the target library.
        native-lib

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        lib_opencv
        lib_dlib)