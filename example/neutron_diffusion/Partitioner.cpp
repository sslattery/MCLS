//---------------------------------------------------------------------------//
/*!
 * \file Partitioner.cpp
 * \author Stuart R. Slattery
 * \brief Mesh partitioner defintion.
 */
//---------------------------------------------------------------------------//

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "Partitioner.hpp"

#include <MCLS_DBC.hpp>

#include <Teuchos_ENull.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace MCLSExamples
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
Partitioner::Partitioner( const RCP_Comm &comm, const RCP_ParameterList &plist )
    : d_num_blocks( SizePair( plist->get<int>( "I_BLOCKS" ), 
			      plist->get<int>( "J_BLOCKS" ) ) )
{
    // Comm parameters.
    unsigned int my_rank = comm->getRank();
    unsigned int my_size = comm->getSize();

    // Check that the block specification and communicator are consistent.
    MCLS_REQUIRE( d_num_blocks.first * d_num_blocks.second == my_size );

    // Block indices.
    int my_j_block = std::floor( my_rank / d_num_blocks.first );
    int my_i_block = my_rank - d_num_blocks.first*my_j_block;
    d_my_blocks.first = my_i_block;   
    d_my_blocks.second = my_j_block;   

    // Uniform grid case.
    std::vector<double> i_edges, j_edges;
    double global_i_min, global_i_max, global_j_min, global_j_max;
    int global_num_i, global_num_j;

    // Get the parameters.
    global_i_min = plist->get<double>( "X_MIN" );
    global_i_max = plist->get<double>( "X_MAX" );
    global_j_min = plist->get<double>( "Y_MIN" );
    global_j_max = plist->get<double>( "Y_MAX" );
    global_num_i = plist->get<int>( "X_NUM_CELLS" );
    global_num_j = plist->get<int>( "Y_NUM_CELLS" );

    // Cell widths.
    double width_i = (global_i_max - global_i_min) / global_num_i;
    double width_j = (global_j_max - global_j_min) / global_num_j;
    d_cell_size.first = width_i;
    d_cell_size.second = width_j;

    // Number of local cells without padding.
    int i_cells_size = std::floor( (double) global_num_i / 
                                   (double) d_num_blocks.first );
    int j_cells_size = std::floor( (double) global_num_j / 
                                   (double) d_num_blocks.second );

    // Remaining cells.
    int i_remainder = global_num_i % d_num_blocks.first;
    int j_remainder = global_num_j % d_num_blocks.second;

    // Start the offset.
    int i_offset = i_cells_size * my_i_block;
    int j_offset = j_cells_size * my_j_block;

    // Pad the number of local cells with the remainder.
    if ( my_i_block < i_remainder )
    {
        ++i_cells_size;
    }
    if ( my_j_block < j_remainder )
    {
        ++j_cells_size;
    }

    // Complete the offset.
    if ( my_i_block < i_remainder )
    {
        i_offset += my_i_block;
    }
    else if ( i_remainder > 0 )
    {
        i_offset += i_remainder;
    }
    if ( my_j_block < j_remainder )
    {
        j_offset += my_j_block;
    }      
    else if ( j_remainder > 0 )
    {
        j_offset += j_remainder;
    }

    // Get the number of vertices.
    int i_edges_size = i_cells_size;
    int j_edges_size = j_cells_size;
    if ( my_i_block == (int) d_num_blocks.first-1 )
    {
        ++i_edges_size;
    }
    if ( my_j_block == (int) d_num_blocks.second-1 )
    {
        ++j_edges_size;
    }

    // Set the local I edges.
    double i_edge_val = width_i * i_offset + global_i_min;
    for ( int i = 0; i < i_edges_size; ++i )
    {
        i_edges.push_back( i_edge_val );
        i_edge_val += width_i;
    }
    if ( my_i_block == (int) d_num_blocks.first-1 )
    {
        MCLS_CHECK( std::abs(global_i_max-i_edges.back()) < 1.0e-6 );
    }

    // Set the local J edges.
    double j_edge_val = width_j * j_offset + global_j_min;
    for ( int j = 0; j < j_edges_size; ++j )
    {
        j_edges.push_back( j_edge_val );
        j_edge_val += width_j;
    }
    if ( my_j_block == (int) d_num_blocks.second-1)
    {
        MCLS_CHECK( std::abs(global_j_max-j_edges.back()) < 1.0e-6 );
    }

    // Set the global I edges.
    i_edge_val = global_i_min;
    for ( int i = 0; i < global_num_i+1; ++i )
    {
        d_global_edges.first.push_back( i_edge_val );
        i_edge_val += width_i;
    }

    // Set the global J edges.
    j_edge_val = global_j_min;
    for ( int j = 0; j < global_num_j+1; ++j )
    {
        d_global_edges.second.push_back( j_edge_val );
        j_edge_val += width_j;
    }

    // Set the local rows.
    int row_idx, idx_i, idx_j;
    for ( int i = 0; i < (int) i_edges.size(); ++i )
    {
        idx_i = i_offset + i;
        d_local_i.push_back( idx_i );
    }
    for ( int j = 0; j < (int) j_edges.size(); ++j )
    {
        idx_j = j_offset + j;
        d_local_j.push_back( idx_j );
    }
    for ( int j = 0; j < (int) j_edges.size(); ++j )
    {
	for ( int i = 0; i < (int) i_edges.size(); ++i )
	{
	    idx_i = i_offset + i;
	    idx_j = j_offset + j;
	    row_idx = idx_i + idx_j*d_global_edges.first.size();
	    d_local_rows.push_back( row_idx );
	}
    }

    // Set the ghost local rows.
    for ( int j = 0; j < (int) j_edges.size(); ++j )
    {
	for ( int i = 0; i < (int) i_edges.size(); ++i )
	{
	    idx_i = i_offset + i;
	    idx_j = j_offset + j;
	    row_idx = idx_i + idx_j*d_global_edges.first.size();
	    d_ghost_local_rows.push_back( row_idx );
	}
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor.
 */
Partitioner::~Partitioner()
{ /* ... */ }

//---------------------------------------------------------------------------//

} 

// end namespace MCLSExamples

//---------------------------------------------------------------------------//
// end Partitioner.cpp
//---------------------------------------------------------------------------//

