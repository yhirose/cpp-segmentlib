add_executable(segmenter
    main.cpp
    predict_command.cpp
    train_command.cpp
)

target_link_libraries(segmenter PRIVATE segmentlib)

# The train subcommand gets its real implementation only when the training
# components are built; otherwise it stays a stub that points at the option.
if(SEGMENTLIB_BUILD_TRAINING)
    target_link_libraries(segmenter PRIVATE segmentlib_train)
    target_compile_definitions(segmenter PRIVATE SEGMENTLIB_HAVE_TRAINING)
endif()
