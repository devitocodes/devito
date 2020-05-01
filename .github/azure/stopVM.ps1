# Stop an (active) VM

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

Write-Output "Stopping VM"
try {
    az vm stop -g $resourceGroupName -n $vmName --verbose
    az vm deallocate -g $resourceGroupName -n $vmName --verbose
    Write-Output "VM stopped"
    }
catch {
    Write-Output "VM not running"
    }
Write-Output ""
