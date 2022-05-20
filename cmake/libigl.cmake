if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 3374c1ad71fab8af2481101b32eab06b93bf81ca
)
FetchContent_MakeAvailable(libigl)