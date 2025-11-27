using YAML

function parse_args()
    config_path = "config/pre_process.yaml"
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--config"
            i == length(ARGS) && error("--config expects a path")
            config_path = ARGS[i + 1]
            i += 1
        else
            @warn "Ignoring unknown argument $arg"
        end
        i += 1
    end
    return config_path
end

function load_config(path::AbstractString)
    println("Loading config from $path")
    return YAML.load_file(path)
end
