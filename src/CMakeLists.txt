add_executable(mini_pytorch3d
    main.cpp
    io.cpp
    render.cpp
    ../cuda/diffrast.cu
)

target_include_directories(mini_pytorch3d
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external
)

target_link_libraries(mini_pytorch3d
    PRIVATE
        ${TORCH_LIBRARIES}
)
