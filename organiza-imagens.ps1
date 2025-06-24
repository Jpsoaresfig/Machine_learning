# Define as classes
$classes = @("Rock", "Paper", "Scissors")

# Cria as pastas se não existirem
foreach ($c in $classes) {
    if (-not (Test-Path $c)) {
        New-Item -ItemType Directory -Name $c | Out-Null
    }
}

# Lê o CSV e copia as imagens para as respectivas pastas
$csvPath = "train\_annotations_filtrado.csv"

Import-Csv -Path $csvPath | ForEach-Object {
    $filename = $_.filename
    $class = $_.class

    if (Test-Path $filename) {
        $dest = ".\$class\$filename"

        # Só copia se ainda não existir
        if (-not (Test-Path $dest)) {
            Copy-Item -Path $filename -Destination $dest
        }
    } else {
        Write-Host "Arquivo não encontrado: $filename"
    }
}
