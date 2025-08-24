#include "BBGameState.h"
#include "Net/UnrealNetwork.h"

ABBGameState::ABBGameState()
{
    PlayerScore = 0;
    GameTime = 0.0f;
}

void ABBGameState::BeginPlay()
{
    Super::BeginPlay();
}

void ABBGameState::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    GameTime += DeltaTime;
}

void ABBGameState::AddScore(int32 ScoreToAdd)
{
    PlayerScore += ScoreToAdd;
}

void ABBGameState::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);
    
    DOREPLIFETIME(ABBGameState, PlayerScore);
    DOREPLIFETIME(ABBGameState, GameTime);
}

void ABBGameState::OnRep_PlayerScore()
{
    // Score updated on clients
}

void ABBGameState::OnRep_GameTime()
{
    // Game time updated on clients
}
