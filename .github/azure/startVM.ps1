# Start a (stopped) VM

[CmdletBinding()]
param(
    [Parameter(Mandatory = $True)]
    [string]
    $servicePrincipal,

    [Parameter(Mandatory = $True)]
    [string]
    $servicePrincipalSecret,

    [Parameter(Mandatory = $True)]
    [string]
    $servicePrincipalTenantId,

    [Parameter(Mandatory = $True)]
    [string]
    $azureSubscriptionName,

    [Parameter(Mandatory = $True)]
    [string]
    $resourceGroupName,

    [Parameter(Mandatory = $True)]
    [string]
    $vmName
)

az login `
    --service-principal `
    --username $servicePrincipal `
    --password $servicePrincipalSecret `
    --tenant $servicePrincipalTenantId

az account set --subscription $azureSubscriptionName

Write-Output "Booting VM"
try {
    az vm start -g $resourceGroupName -n $vmName --verbose
    Write-Output "VM running"
    }
catch {
    Write-Output "VM already running"
    }
# Add pause for safety
Start-Sleep -s 120
az vm list -d -o table --query "[?name=='$vmName']"
Write-Output ""
