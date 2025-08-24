using UnrealBuildTool;

public class BlockBreaker3D : ModuleRules
{
    public BlockBreaker3D(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { 
            "Core", 
            "CoreUObject", 
            "Engine", 
            "InputCore", 
            "HeadMountedDisplay", 
            "NavigationSystem", 
            "AIModule",
            "GameplayTasks",
            "Slate",
            "SlateCore",
            "UMG"
        });
    }
}
