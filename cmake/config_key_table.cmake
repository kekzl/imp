# imp.conf.example -> ${IMP_CONFIG_KEY_TABLE}: one IMP_CONFIG_KEY(section, name) line per key, read
# by tests/test_config_bindings.cpp. Regenerated at configure time when the example changes.

set(_example ${CMAKE_CURRENT_SOURCE_DIR}/imp.conf.example)
set(IMP_CONFIG_KEY_TABLE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(IMP_CONFIG_KEY_TABLE ${IMP_CONFIG_KEY_TABLE_DIR}/config_example_keys.inc)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_example})

file(STRINGS ${_example} _lines REGEX "^(\\[[a-z0-9_]+\\]|[A-Za-z0-9_]+[ \t]*=)")
set(_section "")
set(_table "")
foreach(_line IN LISTS _lines)
    if(_line MATCHES "^\\[([a-z0-9_]+)\\]")
        set(_section ${CMAKE_MATCH_1})
    elseif(_section AND _line MATCHES "^([A-Za-z0-9_]+)[ \t]*=")
        string(APPEND _table "IMP_CONFIG_KEY(${_section}, ${CMAKE_MATCH_1})\n")
    endif()
endforeach()
file(MAKE_DIRECTORY ${IMP_CONFIG_KEY_TABLE_DIR})
file(CONFIGURE OUTPUT ${IMP_CONFIG_KEY_TABLE} CONTENT "${_table}")
